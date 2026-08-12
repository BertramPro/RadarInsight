"""R1: leakage-free, trajectory-level RD-image classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CLASS_NAMES = ["drone", "bird", "balloon", "clutter", "other"]
TARGET_TO_CLASS = {"Drone": 0, "Bird": 1, "Balloon": 2, "Clutter": 3, "Other": 4}
MAT_RE = re.compile(r"^\d+_DAUR_RD_(Drone|Bird|Balloon|Clutter|Other)_(\d+)_(\d+)\.mat$")
COMMON_VR_MIN = -90.0
COMMON_VR_MAX = 89.0
TARGET_VR_WIDTH = 360
DEFAULT_TRAIN_REGISTRY = Path(r"K:\radar\main\data\manifests\rdx_train_registry.json")
REGISTRY_LABEL_TO_CLASS = {
    "DroneTarget": 0,
    "BirdTarget": 1,
    "BalloonTarget": 2,
    "ClutterTarget": 3,
    "UnknownTarget": 4,
}
MINORITY_AUGMENTATION_LABELS = {TARGET_TO_CLASS["Clutter"], TARGET_TO_CLASS["Other"]}


@dataclass(frozen=True)
class Frame:
    path: str
    trajectory_id: str
    source_target: str
    label: int
    augmentation_kind: str = ""
    augmentation_seed: int = 0
    augmentation_source_trajectory_id: str = ""
    augmentation_secondary_path: str = ""
    augmentation_source_trajectory_id_b: str = ""
    augmentation_interpolation_alpha: float = 0.5
    augmentation_cache_plan_key: str = ""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_manifest(dataset_root: Path) -> list[Frame]:
    frames: list[Frame] = []
    for path in sorted((dataset_root / "MAT").glob("*.mat")):
        match = MAT_RE.match(path.name)
        if not match:
            continue
        target, _, trajectory_id = match.groups()
        frames.append(Frame(str(path), trajectory_id, target, TARGET_TO_CLASS[target]))
    if not frames:
        raise RuntimeError(f"No matching MAT files found under {dataset_root / 'MAT'}")
    return frames


def trajectory_table(frames: Iterable[Frame]) -> dict[str, tuple[int, str]]:
    table: dict[str, tuple[int, str]] = {}
    for frame in frames:
        existing = table.setdefault(frame.trajectory_id, (frame.label, frame.source_target))
        if existing != (frame.label, frame.source_target):
            raise ValueError(f"Trajectory {frame.trajectory_id} has inconsistent labels")
    return table


def stratified_trajectory_split(frames: list[Frame], seed: int) -> dict[str, str]:
    table = trajectory_table(frames)
    ids = np.array(sorted(table))
    labels = np.array([table[x][0] for x in ids])
    train_ids, holdout_ids = train_test_split(ids, test_size=0.30, random_state=seed, stratify=labels)
    holdout_labels = np.array([table[x][0] for x in holdout_ids])
    val_ids, test_ids = train_test_split(
        holdout_ids, test_size=0.50, random_state=seed, stratify=holdout_labels
    )
    split = {trajectory_id: "train" for trajectory_id in train_ids}
    split.update({trajectory_id: "val" for trajectory_id in val_ids})
    split.update({trajectory_id: "test" for trajectory_id in test_ids})
    return split


def registry_train_split(frames: list[Frame], registry_path: Path, seed: int) -> tuple[dict[str, str], dict[str, object]]:
    """Use the external train-only registry and stratify its complement into validation/test."""
    try:
        raw = registry_path.read_bytes()
        registry = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read train registry {registry_path}: {exc}") from exc
    if registry.get("scope") != "train_only" or not isinstance(registry.get("samples"), list):
        raise ValueError("Train registry must declare scope='train_only' and contain samples")
    registry_labels: dict[str, int] = {}
    for sample in registry["samples"]:
        try:
            trajectory_id = str(sample["track_id"])
            label = REGISTRY_LABEL_TO_CLASS[str(sample["label_name"])]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed registry sample: {sample}") from exc
        existing = registry_labels.setdefault(trajectory_id, label)
        if existing != label:
            raise ValueError(f"Registry trajectory {trajectory_id} has conflicting labels")
    table = trajectory_table(frames)
    unknown_ids = sorted(set(registry_labels) - set(table))
    if unknown_ids:
        raise ValueError(f"Registry contains {len(unknown_ids)} trajectories absent from RD data (e.g. {unknown_ids[0]})")
    label_mismatches = [trajectory_id for trajectory_id, label in registry_labels.items() if table[trajectory_id][0] != label]
    if label_mismatches:
        raise ValueError(f"Registry labels disagree with RD data (e.g. trajectory {label_mismatches[0]})")
    train_ids = np.array(sorted(registry_labels))
    remaining_ids = np.array(sorted(set(table) - set(registry_labels)))
    remaining_labels = np.array([table[trajectory_id][0] for trajectory_id in remaining_ids])
    if len(remaining_ids) < 2 or len(set(remaining_labels)) < len(CLASS_NAMES):
        raise ValueError("Registry complement cannot support a stratified validation/test split")
    val_ids, test_ids = train_test_split(
        remaining_ids, test_size=0.50, random_state=seed, stratify=remaining_labels
    )
    split = {trajectory_id: "train" for trajectory_id in train_ids}
    split.update({trajectory_id: "val" for trajectory_id in val_ids})
    split.update({trajectory_id: "test" for trajectory_id in test_ids})
    metadata = {
        "mode": "registry_train",
        "registry_path": str(registry_path),
        "registry_version": registry.get("registry_version"),
        "registry_sha256": registry.get("registry_sha256") or hashlib.sha256(raw).hexdigest(),
        "registry_sample_count": len(registry_labels),
        "validation_test_method": "stratified_50_50_complement",
        "validation_test_seed": seed,
    }
    return split, metadata


def split_frames(frames: list[Frame], split: dict[str, str]) -> dict[str, list[Frame]]:
    result: dict[str, list[Frame]] = {"train": [], "val": [], "test": []}
    for frame in frames:
        result[split[frame.trajectory_id]].append(frame)
    return result


def save_manifest(output_dir: Path, frames: list[Frame], split: dict[str, str], split_metadata: dict[str, object]) -> None:
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "path", "trajectory_id", "source_target", "label", "class_name", "split",
            "augmentation_kind", "augmentation_seed", "augmentation_source_trajectory_id",
            "augmentation_secondary_path", "augmentation_source_trajectory_id_b",
            "augmentation_interpolation_alpha",
        ])
        writer.writeheader()
        for frame in frames:
            writer.writerow({**asdict(frame), "class_name": CLASS_NAMES[frame.label], "split": split[frame.trajectory_id]})
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2, ensure_ascii=False)
    write_json(output_dir / "split_source.json", split_metadata)


def sample_training_frames(frames: list[Frame], max_per_trajectory: int, seed: int) -> list[Frame]:
    rng = np.random.default_rng(seed)
    by_trajectory: dict[str, list[Frame]] = defaultdict(list)
    for frame in frames:
        by_trajectory[frame.trajectory_id].append(frame)
    selected: list[Frame] = []
    for trajectory_id, trajectory_frames in sorted(by_trajectory.items()):
        trajectory_frames = sorted(trajectory_frames, key=lambda x: x.path)
        if len(trajectory_frames) <= max_per_trajectory:
            selected.extend(trajectory_frames)
        else:
            indices = rng.choice(len(trajectory_frames), size=max_per_trajectory, replace=False)
            selected.extend(trajectory_frames[index] for index in sorted(indices))
    return selected


def load_rd(path: str, velocity_min: float = COMMON_VR_MIN, velocity_max: float = COMMON_VR_MAX,
            target_width: int = TARGET_VR_WIDTH, resampling: str = "db_linear") -> tuple[np.ndarray, np.ndarray]:
    contents = loadmat(path, variable_names=["data_proc_MTD_result_db", "Vr"])
    rd = contents["data_proc_MTD_result_db"]
    vr = np.asarray(contents["Vr"], dtype=np.float64).reshape(-1)
    if rd.ndim != 2:
        raise ValueError(f"Unexpected RD array shape {rd.shape} in {path}")
    if rd.shape[1] != vr.size:
        raise ValueError(f"RD/Vr mismatch: {rd.shape[1]} columns versus {vr.size} Vr values in {path}")
    if not np.all(np.diff(vr) > 0):
        raise ValueError(f"Vr must be strictly increasing in {path}")
    rd = np.nan_to_num(rd, nan=0.0, posinf=120.0, neginf=0.0)
    target_vr = np.linspace(velocity_min, velocity_max, target_width, dtype=np.float64)
    observed = (target_vr >= vr[0]) & (target_vr <= vr[-1])
    # For the broad-range ablations, files with narrower native Vr ranges are
    # padded outside their measured interval; the mask records those pixels.
    def interpolate(row: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.interp(x, vr, row, left=0.0, right=0.0)
    if resampling == "power_linear":
        power = np.power(10.0, np.clip(rd, -80.0, 120.0) / 10.0)
        sampled = np.stack([interpolate(row, target_vr) for row in power], axis=0)
        normalized = 10.0 * np.log10(np.maximum(sampled, 1e-12))
    elif resampling == "area":
        # Approximate area-preserving bin averaging using midpoint boundaries.
        edges = np.linspace(velocity_min, velocity_max, target_width + 1)
        normalized = np.empty((rd.shape[0], target_width), dtype=np.float64)
        for index in range(target_width):
            left, right = edges[index], edges[index + 1]
            points = np.concatenate(([left], vr[(vr > left) & (vr < right)], [right]))
            values = np.stack([interpolate(row, points) for row in rd], axis=0)
            normalized[:, index] = np.trapz(values, points, axis=1) / max(right - left, 1e-12)
    else:
        # The RD field is dB, so the baseline uses linear interpolation in dB.
        normalized = np.stack([interpolate(row, target_vr) for row in rd], axis=0)
    return normalized.astype(np.float32, copy=False), observed.astype(np.float32)


class RDCache:
    """Read-only, validated cache of velocity-normalized RD frames."""

    def __init__(self, cache_dir: Path, frames: Iterable[Frame], *, velocity_min: float,
                 velocity_max: float, target_width: int, resampling: str,
                 expected_cache_identity: Optional[dict[str, object]] = None,
                 allow_missing: bool = False) -> None:
        self.cache_dir = cache_dir.expanduser().resolve()
        complete = read_json_file(self.cache_dir / "complete.json", {})
        metadata = read_json_file(self.cache_dir / "metadata.json", {})
        index = read_json_file(self.cache_dir / "index.json", {})
        expected = {
            "velocity_min": float(velocity_min), "velocity_max": float(velocity_max),
            "target_width": int(target_width), "resampling": str(resampling),
        }
        actual = metadata.get("preprocessing", {}) if isinstance(metadata, dict) else {}
        if (not isinstance(complete, dict) or complete.get("status") != "complete" or
                not isinstance(index, dict) or actual != expected):
            raise ValueError(f"RD cache is incomplete or incompatible: {self.cache_dir}")
        if expected_cache_identity is not None and metadata.get("cache_identity") != expected_cache_identity:
            raise ValueError(f"RD cache was built for a different split or frame-sampling policy: {self.cache_dir}")
        self.index = {str(path): int(position) for path, position in index.items()}
        missing = [frame.path for frame in frames if frame.path not in self.index]
        if missing and not allow_missing:
            raise ValueError(f"RD cache lacks {len(missing)} requested frames (e.g. {missing[0]})")
        self.allow_missing = bool(allow_missing)
        self.preprocessing = expected
        self.images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
        self.observed = np.load(self.cache_dir / "observed.npy", mmap_mode="r")
        positions = sorted(self.index.values())
        expected_positions = list(range(len(self.index)))
        expected_shape = complete.get("shape")
        if (positions != expected_positions or self.images.ndim != 3 or
                self.images.shape[0] != len(self.index) or self.images.shape[2] != target_width or
                self.observed.shape != (len(self.index), target_width) or
                complete.get("frame_count") != len(self.index) or
                (isinstance(expected_shape, list) and list(self.images.shape) != expected_shape)):
            raise ValueError(f"RD cache arrays have unexpected dimensions: {self.cache_dir}")

    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        position = self.index.get(path)
        if position is None:
            if not self.allow_missing:
                raise KeyError(path)
            return load_rd(
                path,
                self.preprocessing["velocity_min"],
                self.preprocessing["velocity_max"],
                self.preprocessing["target_width"],
                self.preprocessing["resampling"],
            )
        return self.images[position], self.observed[position]


def estimate_normalization(frames: list[Frame], max_frames: int, seed: int, *, velocity_min: float = COMMON_VR_MIN,
                           velocity_max: float = COMMON_VR_MAX, target_width: int = TARGET_VR_WIDTH,
                           resampling: str = "db_linear", rd_cache: Optional[RDCache] = None) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(frames), size=min(max_frames, len(frames)), replace=False)
    sums = 0.0
    squared_sums = 0.0
    count = 0
    for index in indices:
        frame = frames[int(index)]
        array, _ = (rd_cache.load(frame.path) if rd_cache is not None else
                    load_rd(frame.path, velocity_min, velocity_max, target_width, resampling))
        array = np.clip(array, 0.0, 100.0)
        sums += float(array.sum())
        squared_sums += float(np.square(array).sum())
        count += array.size
    mean = sums / count
    std = max((squared_sums / count - mean * mean) ** 0.5, 1e-6)
    return mean, std


class RDDataset(Dataset[tuple[Tensor, int, str]]):
    def __init__(self, frames: list[Frame], mean: float, std: float, *, velocity_min: float = COMMON_VR_MIN,
                  velocity_max: float = COMMON_VR_MAX, target_width: int = TARGET_VR_WIDTH,
                  resampling: str = "db_linear", normalization: str = "global_z", input_mode: str = "rd",
                  augmentation: str = "off", rd_cache: Optional[RDCache] = None,
                  derived_cache_dir: Optional[Path] = None) -> None:
        self.frames = frames
        self.mean = mean
        self.std = std
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max
        self.target_width = target_width
        self.resampling = resampling
        self.normalization = normalization
        self.input_mode = input_mode
        self.augmentation = augmentation
        self.rd_cache = rd_cache
        self.derived_cache_dir = derived_cache_dir.expanduser().resolve() if derived_cache_dir else None
        if self.derived_cache_dir is not None:
            self.derived_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.frames)

    @staticmethod
    def _shift_with_zero_padding(image: np.ndarray, vertical: int, horizontal: int) -> np.ndarray:
        """Translate an RD image without wrapping physical range/Doppler axes."""
        result = np.zeros_like(image)
        source_rows = slice(max(0, -vertical), image.shape[0] - max(0, vertical))
        target_rows = slice(max(0, vertical), image.shape[0] - max(0, -vertical))
        source_columns = slice(max(0, -horizontal), image.shape[1] - max(0, horizontal))
        target_columns = slice(max(0, horizontal), image.shape[1] - max(0, -horizontal))
        result[target_rows, target_columns] = image[source_rows, source_columns]
        return result

    def _augment_minority_rd(self, rd: np.ndarray) -> np.ndarray:
        """Lightweight, physically plausible RD perturbations for scarce classes only."""
        augmented = rd + np.float32(np.random.uniform(-1.5, 1.5))
        augmented = augmented + np.random.normal(0.0, 0.35, size=augmented.shape).astype(np.float32)
        vertical = int(np.random.randint(-1, 2))
        horizontal = int(np.random.randint(-3, 4))
        if vertical or horizontal:
            augmented = self._shift_with_zero_padding(augmented, vertical, horizontal)
        return np.clip(augmented, 0.0, 100.0)

    def _augment_partition_rd(self, rd: np.ndarray, seed: int, path: str) -> np.ndarray:
        """Deterministic track-level perturbation; all frames share offset/shift."""
        rng = np.random.default_rng(int(seed))
        augmented = rd + np.float32(rng.uniform(-1.5, 1.5))
        vertical = int(rng.integers(-1, 2))
        horizontal = int(rng.integers(-3, 4))
        if vertical or horizontal:
            augmented = self._shift_with_zero_padding(augmented, vertical, horizontal)
        # A frame-specific sub-seed keeps noise stable across re-evaluation.
        noise_seed = int.from_bytes(hashlib.sha256(f"{seed}|{path}".encode("utf-8")).digest()[:8], "little") % (2**31 - 1)
        noise = np.random.default_rng(noise_seed).normal(0.0, 0.35, size=rd.shape).astype(np.float32)
        return np.clip(augmented + noise, 0.0, 100.0)

    def _derived_cache_path(self, frame: Frame) -> Optional[Path]:
        if self.derived_cache_dir is None or not frame.augmentation_kind.startswith("partition_rd_"):
            return None

        def version(path_value: str) -> dict[str, object]:
            path = Path(path_value)
            try:
                stat = path.stat()
                return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            except OSError:
                return {"path": str(path), "missing": True}

        identity = {
            "schema": 1,
            "kind": frame.augmentation_kind,
            "seed": int(frame.augmentation_seed),
            "source": version(frame.path),
            "secondary": version(frame.augmentation_secondary_path) if frame.augmentation_secondary_path else None,
            "source_trajectory_id": frame.augmentation_source_trajectory_id,
            "source_trajectory_id_b": frame.augmentation_source_trajectory_id_b,
            "alpha": float(frame.augmentation_interpolation_alpha),
            "augmentation_plan": frame.augmentation_cache_plan_key,
            "preprocessing": {
                "velocity_min": self.velocity_min,
                "velocity_max": self.velocity_max,
                "target_width": self.target_width,
                "resampling": self.resampling,
            },
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        return self.derived_cache_dir / f"{digest}.npz"

    @staticmethod
    def _read_derived_cache(path: Optional[Path]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                return payload["rd"].astype(np.float32, copy=False), payload["observed"].astype(np.float32, copy=False)
        except (OSError, ValueError, KeyError):
            return None

    @staticmethod
    def _write_derived_cache(path: Optional[Path], rd: np.ndarray, observed: np.ndarray) -> None:
        if path is None or path.is_file():
            return
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, rd=rd.astype(np.float32, copy=False),
                                    observed=observed.astype(np.float32, copy=False))
            try:
                os.replace(temporary_name, path)
            except OSError:
                if not path.is_file():
                    raise
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _local_velocity_contrast(self, rd: np.ndarray) -> np.ndarray:
        """Highlight narrow velocity structures against a local velocity baseline."""
        window = max(5, int(round(rd.shape[1] * 0.04)))
        if window % 2 == 0:
            window += 1
        padding = window // 2
        padded = np.pad(rd, ((0, 0), (padding, padding)), mode="edge")
        prefix = np.cumsum(np.pad(padded, ((0, 0), (1, 0)), mode="constant"), axis=1)
        local_mean = (prefix[:, window:] - prefix[:, :-window]) / float(window)
        contrast = (rd - local_mean) / max(self.std, 1e-6)
        return np.clip(contrast, -5.0, 5.0).astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        frame = self.frames[index]
        derived_path = self._derived_cache_path(frame)
        derived = self._read_derived_cache(derived_path)
        if derived is not None:
            rd, observed = derived
        else:
            rd, observed = (self.rd_cache.load(frame.path) if self.rd_cache is not None else
                            load_rd(frame.path, self.velocity_min, self.velocity_max, self.target_width, self.resampling))
        if derived is None and getattr(frame, "augmentation_kind", "") == "partition_rd_smote":
            secondary_path = str(getattr(frame, "augmentation_secondary_path", ""))
            if not secondary_path:
                raise ValueError(f"SMOTE RD frame {frame.trajectory_id} has no secondary source")
            secondary_rd, secondary_observed = (
                self.rd_cache.load(secondary_path) if self.rd_cache is not None else
                load_rd(secondary_path, self.velocity_min, self.velocity_max, self.target_width, self.resampling)
            )
            alpha = float(getattr(frame, "augmentation_interpolation_alpha", 0.5))
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"Invalid RD SMOTE alpha for {frame.trajectory_id}: {alpha}")
            rd = (1.0 - alpha) * rd + alpha * secondary_rd
            observed = np.maximum(observed, secondary_observed)
        rd = np.clip(rd, 0.0, 100.0)
        if derived is None and getattr(frame, "augmentation_kind", "") == "partition_rd_t1":
            rd = self._augment_partition_rd(rd, int(frame.augmentation_seed), frame.path)
        if derived is None and derived_path is not None:
            self._write_derived_cache(derived_path, rd, observed)
        if self.augmentation == "minority_rd" and frame.label in MINORITY_AUGMENTATION_LABELS:
            rd = self._augment_minority_rd(rd)
        physical_rd = rd
        if self.normalization == "frame_z":
            rd = (rd - float(rd.mean())) / max(float(rd.std()), 1e-6)
        elif self.normalization == "frame_robust":
            median = float(np.median(rd)); spread = float(np.percentile(rd, 75) - np.percentile(rd, 25))
            rd = (rd - median) / max(spread, 1e-6)
        elif self.normalization == "minmax":
            low, high = np.percentile(rd, [1, 99]); rd = np.clip((rd - low) / max(high - low, 1e-6), 0.0, 1.0)
        elif self.normalization == "clip":
            rd = rd / 100.0
        else:
            rd = (rd - self.mean) / self.std
        channels = [rd]
        if self.input_mode in {"rd_mask", "rd_peak", "rd_background", "rd_contrast"}:
            if self.input_mode == "rd_mask":
                channels.append(np.broadcast_to(observed[None, :], rd.shape).astype(np.float32))
            elif self.input_mode == "rd_peak":
                peak = int(np.argmax(rd.mean(axis=0)))
                axis = np.arange(rd.shape[1], dtype=np.float32)
                profile = np.exp(-0.5 * ((axis - peak) / max(rd.shape[1] * 0.02, 1.0)) ** 2)
                channels.append(np.broadcast_to(profile[None, :], rd.shape).astype(np.float32))
            elif self.input_mode == "rd_contrast":
                channels.append(self._local_velocity_contrast(physical_rd))
            else:
                background = np.median(rd, axis=1, keepdims=True)
                channels.append((rd - background).astype(np.float32))
        image = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32, copy=False))
        return image, frame.label, frame.trajectory_id


class SmallRDCNN(nn.Module):
    def __init__(self, classes: int = 5, input_channels: int = 1, head: str = "global") -> None:
        super().__init__()
        if head not in {"global", "spatial_2x8"}:
            raise ValueError(f"Unsupported model head: {head}")
        self.head = head
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        pool_size = (1, 1) if head == "global" else (2, 8)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(128 * pool_size[0] * pool_size[1], classes)
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(image)))


def metrics_from_trajectory_probabilities(probs: dict[str, list[np.ndarray]], labels: dict[str, int]) -> dict[str, object]:
    ids = sorted(probs)
    truth = np.array([labels[trajectory_id] for trajectory_id in ids])
    pred = np.array([np.mean(probs[trajectory_id], axis=0).argmax() for trajectory_id in ids])
    confusion_cases: dict[str, list[str]] = defaultdict(list)
    for trajectory_id, actual, predicted in zip(ids, truth, pred):
        if actual != predicted:
            confusion_cases[f"{int(actual)}:{int(predicted)}"].append(trajectory_id)
    return {
        "trajectory_count": len(ids),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=range(5)).tolist(),
        "confusion_cases": dict(confusion_cases),
        "classification_report": classification_report(truth, pred, labels=range(5), target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }


def trajectory_decision_records(probs: dict[str, list[np.ndarray]], labels: dict[str, int]) -> list[dict[str, object]]:
    records = []
    for trajectory_id in sorted(probs, key=str):
        average = np.mean(probs[trajectory_id], axis=0)
        prediction = int(average.argmax())
        truth = int(labels[trajectory_id])
        records.append({
            "trajectory_id": str(trajectory_id),
            "true_class": truth,
            "true_label": CLASS_NAMES[truth],
            "rd_prediction": prediction,
            "rd_prediction_label": CLASS_NAMES[prediction],
            "rd_probabilities": average.tolist(),
            "rd_frame_count": len(probs[trajectory_id]),
        })
    return records


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             progress_dir: Optional[Path] = None, phase: str = "validation",
             epoch: int = 0, total_epochs: int = 1, return_decisions: bool = False) -> dict[str, object] | tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    probs: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, int] = {}
    loss_sum = 0.0
    sample_count = 0
    total_batches = len(loader)
    for batch_index, (image, label, trajectory_ids) in enumerate(loader, start=1):
        output = model(image.to(device, non_blocking=True))
        loss_sum += float(nn.functional.cross_entropy(output, label.to(device, non_blocking=True), reduction="sum").cpu())
        sample_count += label.size(0)
        batch_probs = torch.softmax(output, dim=1).cpu().numpy()
        for trajectory_id, target, prob in zip(trajectory_ids, label.numpy(), batch_probs):
            probs[trajectory_id].append(prob)
            labels[trajectory_id] = int(target)
        if progress_dir is not None:
            write_progress(progress_dir, {"phase": phase, "epoch": epoch, "total_epochs": total_epochs,
                                          "batch": batch_index, "total_batches": total_batches,
                                          "percent": 100.0 * batch_index / max(total_batches, 1),
                                          "loss": loss_sum / max(sample_count, 1)})
    result = metrics_from_trajectory_probabilities(probs, labels)
    result["loss"] = loss_sum / max(sample_count, 1)
    return (result, trajectory_decision_records(probs, labels)) if return_decisions else result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json_file(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def write_progress(output_dir: Path, payload: dict[str, object]) -> None:
    # Publish complete snapshots so the monitor never observes a truncated
    # progress file during a batch update.
    write_json(output_dir / "progress.json", payload)


def random_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: object) -> None:
    if not isinstance(state, dict):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_last_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                         scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int,
                         best_f1: float, best_epoch: int, stale_epochs: int,
                         history: list[dict[str, object]], mean: float, std: float) -> None:
    payload = {
        "checkpoint_version": 2,
        "checkpoint_type": "last_completed_epoch",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "mean": mean,
        "std": std,
        "class_names": CLASS_NAMES,
        "random_state": random_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-frames-per-trajectory", type=int, default=32)
    parser.add_argument("--norm-samples", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--velocity-min", type=float, default=COMMON_VR_MIN)
    parser.add_argument("--velocity-max", type=float, default=COMMON_VR_MAX)
    parser.add_argument("--target-width", type=int, default=TARGET_VR_WIDTH)
    parser.add_argument("--resampling", choices=["db_linear", "power_linear", "area"], default="db_linear")
    parser.add_argument("--normalization", choices=["global_z", "frame_z", "frame_robust", "minmax", "clip"], default="global_z")
    parser.add_argument("--input-mode", choices=["rd", "rd_mask", "rd_peak", "rd_background", "rd_contrast"], default="rd")
    parser.add_argument("--split-mode", choices=["fixed_grouped", "random_stratified", "registry_train"], default="registry_train")
    parser.add_argument("--train-registry", type=Path, default=DEFAULT_TRAIN_REGISTRY)
    parser.add_argument("--grouped-split", type=Path, default=None,
                        help="Explicit trajectory split; may contain a strict subset of the dataset for OOF training")
    parser.add_argument("--model-head", choices=["global", "spatial_2x8"], default="global")
    parser.add_argument("--augmentation", choices=["off", "minority_rd"], default="off")
    parser.add_argument("--rd-cache", type=Path, default=None,
                        help="Completed local RD preprocessing cache compatible with this velocity setup")
    parser.add_argument("--allow-cache-misses", action="store_true",
                        help="Preprocess frames absent from the cache; intended for OOF fold training")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from last.pt, or warm-resume a legacy run from best.pt")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--partition-augmentation-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition-augmentation-targets-train", type=int, nargs=5, default=None)
    parser.add_argument("--partition-augmentation-targets-val", type=int, nargs=5, default=None)
    parser.add_argument("--partition-augmentation-targets-test", type=int, nargs=5, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A scheduler-managed exact resume deliberately reuses the original
    # experiment directory so its identity, logs and experiment number stay
    # intact.  Fresh runs must still reject a non-empty output directory.
    resume_in_place = (args.resume is not None and args.resume.expanduser().resolve().parent == args.output_dir.resolve())
    if any(args.output_dir.iterdir()) and not resume_in_place:
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    write_progress(args.output_dir, {"phase": "preparing", "stage": "loading_manifest", "epoch": 0,
                                     "total_epochs": args.epochs, "batch": 0, "total_batches": 0,
                                     "percent": 0.0})
    resume_path = args.resume.expanduser().resolve() if args.resume is not None else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    resume_checkpoint = (torch.load(resume_path, map_location="cpu", weights_only=False)
                         if resume_path is not None else None)
    if resume_checkpoint is not None and "model_state" not in resume_checkpoint:
        raise ValueError(f"Resume checkpoint has no model_state: {resume_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = build_manifest(args.dataset_root)
    write_progress(args.output_dir, {"phase": "preparing", "stage": "building_split", "epoch": 0,
                                     "total_epochs": args.epochs, "batch": 0, "total_batches": 0,
                                     "percent": 0.0})
    source_dir = resume_path.parent if resume_path is not None else None
    source_split = read_json_file(source_dir / "split.json", {}) if source_dir is not None else {}
    if isinstance(source_split, dict) and source_split:
        split = {str(key): str(value) for key, value in source_split.items()}
        frame_trajectory_ids = set(trajectory_table(frames))
        if set(split) != frame_trajectory_ids:
            raise ValueError("Resume split.json does not match the current dataset trajectories")
        source_config = read_json_file(source_dir / "config.json", {})
        split_metadata = read_json_file(source_dir / "split_source.json", {})
        if not isinstance(split_metadata, dict) or not split_metadata:
            split_metadata = source_config.get("split", {}) if isinstance(source_config, dict) else {}
        if not isinstance(split_metadata, dict) or not split_metadata:
            split_metadata = {"mode": args.split_mode, "seed": args.seed}
    elif args.grouped_split is not None:
        grouped_split_path = args.grouped_split.expanduser().resolve()
        raw_split = read_json_file(grouped_split_path, {})
        if not isinstance(raw_split, dict) or not raw_split:
            raise ValueError(f"Grouped split is empty or invalid: {grouped_split_path}")
        # Accept both the RD-native ``trajectory_id -> partition`` mapping
        # and the authoritative F manifest used by the TR/fusion branch.
        # The latter stores the three disjoint trajectory-id lists under
        # ``*_group_ids``.
        group_keys = {
            "train": "train_group_ids",
            "val": "val_group_ids",
            "test": "test_group_ids",
        }
        if all(key in raw_split for key in group_keys.values()):
            split = {
                str(trajectory_id): partition
                for partition, key in group_keys.items()
                for trajectory_id in raw_split[key]
            }
        else:
            split = {str(key): str(value) for key, value in raw_split.items()}
        unsupported = set(split.values()) - {"train", "val", "test"}
        if unsupported:
            raise ValueError(f"Grouped split contains unsupported partitions: {sorted(unsupported)}")
        dataset_ids = set(trajectory_table(frames))
        missing_ids = sorted(set(split) - dataset_ids)
        if missing_ids:
            raise ValueError(f"Grouped split contains trajectories absent from RD data (e.g. {missing_ids[0]})")
        frames = [frame for frame in frames if frame.trajectory_id in split]
        split_metadata = {
            # Keep the public split-mode name for the authoritative F split.
            # This must match the cache builder's identity exactly; arbitrary
            # externally supplied mappings remain labelled grouped_split.
            "mode": "fixed_grouped" if args.split_mode == "fixed_grouped" else "grouped_split",
            "manifest": str(grouped_split_path),
            "sha256": hashlib.sha256(grouped_split_path.read_bytes()).hexdigest(),
            "trajectory_count": len(split),
        }
    elif args.split_mode == "registry_train":
        split, split_metadata = registry_train_split(frames, args.train_registry, args.seed)
    elif args.split_mode == "fixed_grouped":
        raise ValueError("fixed_grouped requires --grouped-split")
    else:
        split = stratified_trajectory_split(frames, args.seed)
        split_metadata = {"mode": "random_stratified", "train_validation_test_ratio": [0.70, 0.15, 0.15], "seed": args.seed}
    save_manifest(args.output_dir, frames, split, split_metadata)
    partitions = split_frames(frames, split)
    from radar_fusion.partition_augmentation import expand_rd_frames, validate_targets
    train_targets = validate_targets(args.partition_augmentation_targets_train, partitions["train"])
    val_targets = validate_targets(args.partition_augmentation_targets_val, partitions["val"])
    test_targets = validate_targets(args.partition_augmentation_targets_test, partitions["test"])
    partition_augments = {"enabled": bool(args.partition_augmentation_diagnostics),
                          "description": "validation/test related virtual trajectory diagnostic; not independent samples"}
    # The train target is the sole training-copy mechanism.  Do not tie it to
    # the validation/test diagnostic switch, otherwise a UI toggle silently
    # changes the model's training distribution.
    train_augmented_frames, train_aug_manifest = expand_rd_frames(partitions["train"], partition="train", targets=train_targets, seed=args.seed)
    selected_train = sample_training_frames(train_augmented_frames, args.max_train_frames_per_trajectory, args.seed)
    partition_augments["train"] = train_aug_manifest
    if args.partition_augmentation_diagnostics:
        val_augmented_frames, val_aug_manifest = expand_rd_frames(partitions["val"], partition="val", targets=val_targets, seed=args.seed)
        test_augmented_frames, test_aug_manifest = expand_rd_frames(partitions["test"], partition="test", targets=test_targets, seed=args.seed)
        partition_augments.update({"val": val_aug_manifest, "test": test_aug_manifest})
    else:
        val_augmented_frames = partitions["val"]
        test_augmented_frames = partitions["test"]
    write_progress(args.output_dir, {"phase": "preparing", "stage": "preprocessing_rd", "epoch": 0,
                                     "total_epochs": args.epochs, "batch": 0, "total_batches": 0,
                                     "percent": 0.0})
    # Cache identity is deliberately tied to original, physical frames.  A
    # virtual track reuses its source MAT frame and is perturbed in Dataset,
    # so adding its duplicate path here would not create a cache entry.  Test
    # frames stay out of the training cache and are deterministically loaded
    # on demand when an explicit test run is requested.
    cache_frames = [*partitions["train"], *partitions["val"]]
    cache_identity = {
        "split": split_metadata,
        "max_train_frames_per_trajectory": int(args.max_train_frames_per_trajectory),
        "include_test": False,
    }
    rd_cache = (RDCache(args.rd_cache, cache_frames, velocity_min=args.velocity_min,
                        velocity_max=args.velocity_max, target_width=args.target_width,
                        resampling=args.resampling, expected_cache_identity=cache_identity,
                        # Train virtual tracks share paths with their source
                        # frames.  Permit those aliases and any explicit test
                        # frames to take the exact same preprocessing path on
                        # a cache miss, instead of making virtual augmentation
                        # unusable with the recommended cache.
                        allow_missing=True)
                if args.rd_cache is not None else None)
    if resume_checkpoint is not None and resume_checkpoint.get("mean") is not None and resume_checkpoint.get("std") is not None:
        mean, std = float(resume_checkpoint["mean"]), float(resume_checkpoint["std"])
    else:
        mean, std = estimate_normalization(selected_train, args.norm_samples, args.seed,
                                           velocity_min=args.velocity_min, velocity_max=args.velocity_max,
                                           target_width=args.target_width, resampling=args.resampling,
                                           rd_cache=rd_cache)
    configuration = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()} | {"device": str(device), "normalization_mean": mean, "normalization_std": std,
                                  "velocity_preprocessing": {"source_coordinate": "Vr", "common_interval_mps": [args.velocity_min, args.velocity_max],
                                                             "target_width": args.target_width, "interpolation": args.resampling},
                                  "all_frame_counts": {name: len(value) for name, value in partitions.items()},
                                  "train_frame_count_after_sampling": len(selected_train),
                                  "trajectory_counts": dict(Counter(split.values())), "split": split_metadata,
                                  "train_class_frame_counts": {CLASS_NAMES[label]: count for label, count in sorted(Counter(frame.label for frame in selected_train).items())},
                                  "augmentation": {"mode": args.augmentation,
                                                   "classes": [CLASS_NAMES[label] for label in sorted(MINORITY_AUGMENTATION_LABELS)] if args.augmentation == "minority_rd" else []}}
    configuration["partition_augmentation_diagnostics"] = partition_augments
    if rd_cache is not None:
        configuration["rd_cache"] = str(rd_cache.cache_dir)
    if resume_path is not None:
        configuration["resume"] = {"checkpoint": str(resume_path), "source_output_dir": str(source_dir)}
    configuration["dataset_root"] = str(args.dataset_root)
    configuration["output_dir"] = str(args.output_dir)
    write_json(args.output_dir / "config.json", configuration)
    write_json(args.output_dir / "partition_augmentation_manifest.json", partition_augments)
    print(json.dumps(configuration, ensure_ascii=False, indent=2), flush=True)

    dataset_options = {"velocity_min": args.velocity_min, "velocity_max": args.velocity_max,
                       "target_width": args.target_width, "resampling": args.resampling,
                       "normalization": args.normalization, "input_mode": args.input_mode,
                       "augmentation": args.augmentation}
    train_dataset = RDDataset(selected_train, mean, std, rd_cache=rd_cache, **dataset_options)
    eval_dataset_options = dict(dataset_options)
    eval_dataset_options["augmentation"] = "off"
    val_dataset = RDDataset(partitions["val"], mean, std, rd_cache=rd_cache, **eval_dataset_options)
    derived_cache_dir = Path(__file__).resolve().parents[1] / "cache" / "rd_partition_augmented"
    val_augmented_dataset = RDDataset(
        val_augmented_frames, mean, std, rd_cache=rd_cache,
        derived_cache_dir=derived_cache_dir, **eval_dataset_options,
    )
    class_counts = Counter(frame.label for frame in selected_train)
    weights = torch.tensor([1.0 / class_counts[frame.label] for frame in selected_train], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(selected_train), replacement=True)
    loader_kwargs = {"num_workers": args.workers, "pin_memory": device.type == "cuda", "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
    eval_loader_kwargs = {"num_workers": args.workers, "pin_memory": device.type == "cuda", "persistent_workers": args.workers > 0}
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **eval_loader_kwargs)
    val_augmented_loader = DataLoader(val_augmented_dataset, batch_size=args.batch_size, shuffle=False, **eval_loader_kwargs)
    test_loader = None
    if not args.skip_test:
        test_dataset = RDDataset(partitions["test"], mean, std, rd_cache=rd_cache, **eval_dataset_options)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **eval_loader_kwargs)

    input_channels = 1 if args.input_mode == "rd" else 2
    model = SmallRDCNN(input_channels=input_channels, head=args.model_head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_augmented_records: list[dict[str, object]] = []
    stale_epochs = 0
    start_epoch = 1
    resume_mode = None
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        source_history = resume_checkpoint.get("history")
        if not isinstance(source_history, list):
            source_history = read_json_file(source_dir / "history.json", [])
        history = source_history if isinstance(source_history, list) else []
        last_history_epoch = max((int(item.get("epoch", 0)) for item in history if isinstance(item, dict)), default=0)
        if resume_checkpoint.get("optimizer_state") is not None and resume_checkpoint.get("scheduler_state") is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            completed_epoch = int(resume_checkpoint.get("epoch", last_history_epoch))
            best_f1 = float(resume_checkpoint.get("best_val_macro_f1", -1.0))
            best_epoch = int(resume_checkpoint.get("best_epoch", 0))
            stale_epochs = int(resume_checkpoint.get("stale_epochs", max(0, completed_epoch - best_epoch)))
            restore_random_state(resume_checkpoint.get("random_state"))
            resume_mode = "exact_last_epoch"
        else:
            # Legacy runs only saved the best model. Preserve their completed
            # history, LR position and early-stopping counter while warming up
            # a fresh optimizer from that best model.
            completed_epoch = max(last_history_epoch, int(resume_checkpoint.get("epoch", 0)))
            best_item = max(history, key=lambda item: item.get("val_trajectory_macro_f1", float("-inf")), default={})
            best_f1 = float(resume_checkpoint.get("best_val_macro_f1", best_item.get("val_trajectory_macro_f1", -1.0)))
            best_epoch = int(resume_checkpoint.get("epoch", best_item.get("epoch", 0)))
            stale_epochs = max(0, completed_epoch - best_epoch)
            cosine_lr = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * completed_epoch / args.epochs))
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr
            scheduler.last_epoch = completed_epoch
            scheduler._step_count = completed_epoch + 1
            scheduler._last_lr = [cosine_lr for _ in optimizer.param_groups]
            resume_mode = "legacy_best_warm_start"
        start_epoch = completed_epoch + 1
        source_best = source_dir / "best.pt"
        if source_best.is_file():
            shutil.copy2(source_best, args.output_dir / "best.pt")
        else:
            shutil.copy2(resume_path, args.output_dir / "best.pt")
        source_validation = source_dir / "validation_best.json"
        if source_validation.is_file():
            shutil.copy2(source_validation, args.output_dir / "validation_best.json")
        source_validation_latest = source_dir / "validation_latest.json"
        if source_validation_latest.is_file():
            shutil.copy2(source_validation_latest, args.output_dir / "validation_latest.json")
        configuration["resume"].update({"mode": resume_mode, "completed_epoch": completed_epoch,
                                         "next_epoch": start_epoch, "best_epoch": best_epoch})
        write_json(args.output_dir / "config.json", configuration)
        write_json(args.output_dir / "history.json", history)
        print(json.dumps({"resume_mode": resume_mode, "checkpoint": str(resume_path),
                          "completed_epoch": completed_epoch, "next_epoch": start_epoch,
                          "best_epoch": best_epoch, "best_val_macro_f1": best_f1}, ensure_ascii=False), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        total_batches = len(train_loader)
        write_progress(args.output_dir, {"phase": "train", "epoch": epoch, "total_epochs": args.epochs,
                                         "batch": 0, "total_batches": total_batches,
                                         "percent": (epoch - 1) / args.epochs * 100})
        for batch_index, (image, label, _) in enumerate(train_loader, start=1):
            image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * label.size(0)
            correct += int((logits.argmax(1) == label).sum())
            count += label.size(0)
            write_progress(args.output_dir, {"phase": "train", "epoch": epoch, "total_epochs": args.epochs,
                                             "batch": batch_index, "total_batches": total_batches,
                                             "percent": ((epoch - 1) + batch_index / total_batches) / args.epochs * 100,
                                             "train_loss": loss_sum / count,
                                             "train_frame_accuracy": correct / count})
        validation = evaluate(model, val_loader, device, args.output_dir, "validation", epoch, args.epochs)
        validation_augmented = None
        validation_augmented_records: list[dict[str, object]] = []
        if args.partition_augmentation_diagnostics:
            validation_augmented, validation_augmented_records = evaluate(model, val_augmented_loader, device, args.output_dir, "validation_augmented", epoch, args.epochs, True)
            write_json(args.output_dir / "validation_augmented_latest.json", validation_augmented | {"epoch": epoch})
        # Persist the current trajectory-level errors on every validation pass.
        # This survives an interrupted run, while validation_best.json remains
        # the immutable record for the best checkpoint.
        write_json(args.output_dir / "validation_latest.json", validation | {"epoch": epoch})
        result = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / count,
                  "train_frame_accuracy": correct / count, "val_loss": validation["loss"],
                  "val_trajectory_macro_f1": validation["macro_f1"],
                  "val_trajectory_accuracy": validation["accuracy"]}
        if validation_augmented is not None:
            result.update({"val_augmented_loss": validation_augmented["loss"],
                           "val_augmented_trajectory_macro_f1": validation_augmented["macro_f1"],
                           "val_augmented_trajectory_accuracy": validation_augmented["accuracy"]})
        history.append(result)
        write_json(args.output_dir / "history.json", history)
        write_progress(args.output_dir, {"phase": "validation", "epoch": epoch, "total_epochs": args.epochs,
                                         "batch": total_batches, "total_batches": total_batches,
                                         "percent": epoch / args.epochs * 100,
                                         "train_loss": result["train_loss"],
                                         "train_frame_accuracy": result["train_frame_accuracy"],
                                         "val_loss": result["val_loss"],
                                         "val_trajectory_macro_f1": result["val_trajectory_macro_f1"],
                                         "val_trajectory_accuracy": result["val_trajectory_accuracy"]})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if validation["macro_f1"] > best_f1:
            best_f1 = float(validation["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_val_macro_f1": best_f1,
                        "mean": mean, "std": std, "class_names": CLASS_NAMES}, args.output_dir / "best.pt")
            write_json(args.output_dir / "validation_best.json", validation)
            if validation_augmented is not None:
                write_json(args.output_dir / "validation_augmented_best.json", validation_augmented)
            best_augmented_records = validation_augmented_records
        else:
            stale_epochs += 1
        scheduler.step()
        save_last_checkpoint(args.output_dir / "last.pt", model=model, optimizer=optimizer,
                             scheduler=scheduler, epoch=epoch, best_f1=best_f1,
                             best_epoch=best_epoch, stale_epochs=stale_epochs,
                             history=history, mean=mean, std=std)
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation Macro-F1={best_f1:.4f}", flush=True)
            break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if args.partition_augmentation_diagnostics and not best_augmented_records:
        _, best_augmented_records = evaluate(model, val_augmented_loader, device, args.output_dir, "validation_augmented", best_epoch or args.epochs, args.epochs, True)
    if args.partition_augmentation_diagnostics:
        (args.output_dir / "trajectory_decisions_augmented.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in best_augmented_records), encoding="utf-8"
        )
    if args.skip_test:
        write_json(args.output_dir / "ablation_complete.json", {"best_epoch": checkpoint["epoch"],
                                                                  "best_val_macro_f1": checkpoint["best_val_macro_f1"],
                                                                  "test_deferred": True})
        print(json.dumps({"best_val_macro_f1": checkpoint["best_val_macro_f1"], "test_deferred": True}, ensure_ascii=False), flush=True)
    else:
        test_result = evaluate(model, test_loader, device, args.output_dir, "testing", args.epochs, args.epochs)
        write_json(args.output_dir / "test_trajectory_metrics.json", test_result)
        if args.partition_augmentation_diagnostics:
            test_augmented_dataset = RDDataset(
                test_augmented_frames, mean, std, rd_cache=rd_cache,
                derived_cache_dir=derived_cache_dir, **eval_dataset_options,
            )
            test_augmented_loader = DataLoader(test_augmented_dataset, batch_size=args.batch_size, shuffle=False, **eval_loader_kwargs)
            test_augmented_result, test_augmented_records = evaluate(model, test_augmented_loader, device, args.output_dir, "testing_augmented", args.epochs, args.epochs, True)
            write_json(args.output_dir / "test_augmented_trajectory_metrics.json", test_augmented_result)
            (args.output_dir / "trajectory_decisions_test_augmented.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in test_augmented_records), encoding="utf-8"
            )
        print(json.dumps({"test_trajectory_macro_f1": test_result["macro_f1"], "test_trajectory_accuracy": test_result["accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

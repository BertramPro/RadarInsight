"""Render class-aggregated RD structures and velocity profiles for the validation split."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat

CLASS_NAMES = ["drone", "bird", "balloon", "clutter", "other"]
CLASS_NAMES_ZH = ["无人机", "飞鸟", "空飘球", "杂波", "未知目标"]
TARGET_TO_CLASS = {"Drone": 0, "Bird": 1, "Balloon": 2, "Clutter": 3, "Other": 4}


@dataclasses.dataclass(frozen=True)
class Frame:
    path: str
    trajectory_id: str
    label: int


def build_manifest(dataset_root: Path) -> list[Frame]:
    frames: list[Frame] = []
    for path in sorted((dataset_root / "MAT").glob("*.mat")):
        parts = path.stem.split("_")
        if len(parts) != 6 or parts[1:3] != ["DAUR", "RD"] or parts[3] not in TARGET_TO_CLASS:
            continue
        frames.append(Frame(str(path), parts[5], TARGET_TO_CLASS[parts[3]]))
    if not frames:
        raise RuntimeError(f"No matching MAT files found under {dataset_root / 'MAT'}")
    return frames


def split_frames(frames: list[Frame], split: dict[str, str]) -> dict[str, list[Frame]]:
    result: dict[str, list[Frame]] = {"train": [], "val": [], "test": []}
    for frame in frames:
        result[split[frame.trajectory_id]].append(frame)
    return result


class RDCache:
    """Read the existing velocity-normalized RD cache without training dependencies."""

    def __init__(self, cache_dir: Path, frames: list[Frame], *, velocity_min: float,
                 velocity_max: float, target_width: int, resampling: str) -> None:
        metadata = read_json(cache_dir / "metadata.json")
        index = read_json(cache_dir / "index.json")
        expected = {
            "velocity_min": float(velocity_min), "velocity_max": float(velocity_max),
            "target_width": int(target_width), "resampling": str(resampling),
        }
        if metadata.get("preprocessing") != expected:
            raise ValueError(f"RD cache preprocessing does not match: {cache_dir}")
        self.index = {str(path): int(position) for path, position in index.items()}
        missing = [frame.path for frame in frames if frame.path not in self.index]
        if missing:
            raise ValueError(f"RD cache lacks {len(missing)} requested frames (e.g. {missing[0]})")
        self.images = np.load(cache_dir / "images.npy", mmap_mode="r")

    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        return self.images[self.index[path]], np.empty(0, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-min", type=float, default=-15.0)
    parser.add_argument("--focus-max", type=float, default=15.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def fwhm_width(velocity: np.ndarray, profile: np.ndarray) -> float:
    baseline = float(np.percentile(profile, 20.0))
    threshold = baseline + 0.5 * max(float(profile.max()) - baseline, 1e-12)
    peak = int(profile.argmax())
    active = profile >= threshold
    left = right = peak
    while left > 0 and active[left - 1]:
        left -= 1
    while right + 1 < len(profile) and active[right + 1]:
        right += 1
    return float(velocity[right] - velocity[left])


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config = read_json(checkpoint.parent / "config.json")
    frames = build_manifest(Path(str(config["dataset_root"])))
    split = read_json(checkpoint.parent / "split.json")
    val_frames = split_frames(frames, split)["val"]
    cache = RDCache(Path(str(config["rd_cache"])), val_frames,
                    velocity_min=float(config["velocity_min"]), velocity_max=float(config["velocity_max"]),
                    target_width=int(config["target_width"]), resampling=str(config["resampling"]))
    trajectories = defaultdict(list)
    labels = {}
    for frame in val_frames:
        trajectories[frame.trajectory_id].append(frame)
        labels[frame.trajectory_id] = frame.label
    class_maps = defaultdict(list)
    for trajectory_id, trajectory_frames in trajectories.items():
        maps = [np.clip(np.asarray(cache.load(frame.path)[0], dtype=np.float32), 0.0, 100.0)
                for frame in trajectory_frames]
        trajectory_mean = np.mean(maps, axis=0)
        # Remove the local stationary background for visualization only; each
        # trajectory remains equally weighted in the class median below.
        relative = trajectory_mean - np.median(trajectory_mean, axis=1, keepdims=True)
        class_maps[labels[trajectory_id]].append(relative)

    velocity = np.linspace(float(config["velocity_min"]), float(config["velocity_max"]),
                           int(config["target_width"]))
    focused = (velocity >= args.focus_min) & (velocity <= args.focus_max)
    structures = {label: np.median(np.stack(class_maps[label], axis=0), axis=0)
                  for label in range(len(CLASS_NAMES))}
    displayed = np.concatenate([structure[:, focused].reshape(-1) for structure in structures.values()])
    vmin = float(np.percentile(displayed, 2.0))
    vmax = float(np.percentile(displayed, 98.0))
    profiles = {label: np.maximum(structure.max(axis=0), 0.0) for label, structure in structures.items()}
    profile_norm = {label: value / max(float(value.max()), 1e-12) for label, value in profiles.items()}
    widths = {label: fwhm_width(velocity, profile_norm[label]) for label in range(len(CLASS_NAMES))}

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    figure, axes = plt.subplots(2, 5, figsize=(17.2, 5.8), sharex="col",
                                gridspec_kw={"height_ratios": [3.5, 1.35]}, constrained_layout=True)
    artist = None
    for label in range(len(CLASS_NAMES)):
        top, bottom = axes[0, label], axes[1, label]
        artist = top.imshow(structures[label][:, focused], origin="lower", aspect="auto", cmap="magma",
                            vmin=vmin, vmax=vmax,
                            extent=(args.focus_min, args.focus_max, 0, structures[label].shape[0] - 1))
        top.set_title(CLASS_NAMES_ZH[label])
        top.set_ylabel("相对距离单元" if label == 0 else "")
        top.set_xlim(args.focus_min, args.focus_max)
        local_velocity = velocity[focused]
        local_profile = profile_norm[label][focused]
        bottom.plot(local_velocity, local_profile, color="#c65102", linewidth=1.8)
        bottom.fill_between(local_velocity, 0.0, local_profile, color="#c65102", alpha=0.18)
        bottom.axhline(0.5, color="#75808c", linewidth=0.8, linestyle="--")
        bottom.set_ylim(0.0, 1.05)
        bottom.set_xlim(args.focus_min, args.focus_max)
        bottom.set_xlabel("径向速度/(m·s$^{-1}$)")
        bottom.set_ylabel("归一化响应" if label == 0 else "")
        bottom.text(0.03, 0.88, f"全速度半峰宽度 {widths[label]:.2f} m/s",
                    transform=bottom.transAxes, fontsize=8, va="top")
    colorbar = figure.colorbar(artist, ax=axes[0, :].tolist(), shrink=0.84, pad=0.015)
    colorbar.set_label("相对行背景幅度/dB")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "five_class_aggregate_rd_structure.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    manifest = {
        "checkpoint": str(checkpoint), "split": "validation", "trajectory_count": len(trajectories),
        "aggregation": "Per trajectory: mean RD over frames, then subtract the per-range-row median. Per class: median over trajectories with equal trajectory weights.",
        "focus_velocity_mps": [args.focus_min, args.focus_max],
        "color_scale_relative_db": [vmin, vmax],
        "class_trajectory_counts": {CLASS_NAMES[label]: len(class_maps[label]) for label in range(len(CLASS_NAMES))},
        "full_velocity_profile_fwhm_mps": {CLASS_NAMES[label]: widths[label] for label in range(len(CLASS_NAMES))},
    }
    (output / "five_class_aggregate_rd_structure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

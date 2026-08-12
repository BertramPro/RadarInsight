"""Trajectory data adapters and fixed grouped-split handling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .track_features import encode_track


LABEL_TO_CLASS = {
    "DroneTarget": 0,
    "BirdTarget": 1,
    "BalloonTarget": 2,
    "ClutterTarget": 3,
    "UnknownTarget": 4,
}


def load_grouped_split(path: Path | str, *, allow_subset: bool = False) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if all(key in payload for key in ("train_group_ids", "val_group_ids", "test_group_ids")):
        # Do not let a malformed manifest silently assign one trajectory to
        # whichever partition happens to be processed last.  That would turn
        # a split typo into an undetected train/validation/test leakage risk.
        partition_ids = {
            partition: [str(value) for value in payload[key]]
            for partition, key in (
                ("train", "train_group_ids"),
                ("val", "val_group_ids"),
                ("test", "test_group_ids"),
            )
        }
        seen: dict[str, str] = {}
        duplicate_within: dict[str, list[str]] = {}
        duplicate_across: dict[str, list[str]] = {}
        for partition, values in partition_ids.items():
            for trajectory_id in values:
                previous = seen.get(trajectory_id)
                if previous is None:
                    seen[trajectory_id] = partition
                elif previous == partition:
                    duplicate_within.setdefault(trajectory_id, [partition]).append(partition)
                else:
                    duplicate_across.setdefault(trajectory_id, [previous]).append(partition)
        if duplicate_within or duplicate_across:
            examples = list(duplicate_across)[:3] or list(duplicate_within)[:3]
            detail = ", ".join(
                f"{trajectory_id}: {duplicate_across.get(trajectory_id) or duplicate_within[trajectory_id]}"
                for trajectory_id in examples
            )
            raise ValueError(f"grouped split contains duplicate trajectory assignments ({detail})")
        split = {
            str(trajectory_id): partition
            for partition, values in partition_ids.items()
            for trajectory_id in values
        }
    elif payload and all(value in {"train", "val", "test"} for value in payload.values()):
        split = {str(key): str(value) for key, value in payload.items()}
    else:
        raise ValueError(f"unsupported grouped split format: {path}")
    if allow_subset and not 0 < len(split) <= 1549:
        raise ValueError(f"expected a non-empty CQ-08 subset of at most 1549 trajectories, found {len(split)}")
    if not allow_subset and len(split) != 1549:
        raise ValueError(f"expected 1549 CQ-08 trajectories, found {len(split)}")
    return split


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    csv_path: Path
    label: int
    height_missing: bool
    phase_missing: bool
    augmentation_kind: str = ""
    augmentation_seed: int = 0
    augmentation_source_trajectory_id: str = ""
    augmentation_frame_drop_fraction: float = 0.10
    augmentation_amplitude_scale_max_delta: float = 0.05
    augmentation_snr_offset_db: float = 1.0
    augmentation_secondary_csv_path: Path | None = None
    augmentation_source_trajectory_id_b: str = ""
    augmentation_interpolation_alpha: float = 0.5


@dataclass(frozen=True)
class TrajectoryItem:
    trajectory_id: str
    sequence: Tensor
    physical: Tensor
    label: int


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_trajectory_records(index_path: Path | str, split: dict[str, str], partition: str) -> list[TrajectoryRecord]:
    index_path = Path(index_path).expanduser().resolve()
    project_root = index_path.parents[2]
    frame = pd.read_csv(index_path)
    frame = frame[frame["source_name"].astype(str).eq("cq08_track")].copy()
    records: list[TrajectoryRecord] = []
    seen: set[str] = set()
    for row in frame.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        prefix = "cq08|track|"
        if not sample_id.startswith(prefix):
            continue
        trajectory_id = sample_id[len(prefix):]
        if split.get(trajectory_id) != partition:
            continue
        if trajectory_id in seen:
            raise ValueError(f"duplicate trajectory in track index: {trajectory_id}")
        seen.add(trajectory_id)
        label_name = str(row["label_name"])
        if label_name not in LABEL_TO_CLASS:
            raise ValueError(f"unsupported CQ-08 label {label_name!r} for trajectory {trajectory_id}")
        csv_path = Path(str(row["csv_path"]))
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        records.append(
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                csv_path=csv_path,
                label=LABEL_TO_CLASS[label_name],
                height_missing=_as_bool(row.get("height_missing", True)),
                phase_missing=_as_bool(row.get("rd_phase_missing", True)),
            )
        )
    expected = {trajectory_id for trajectory_id, value in split.items() if value == partition}
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"track index lacks {len(missing)} {partition} trajectories (e.g. {missing[0]})")
    return sorted(records, key=lambda record: int(record.trajectory_id))


class TrajectoryDataset(Dataset[TrajectoryItem]):
    def __init__(self, records: list[TrajectoryRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrajectoryItem:
        record = self.records[index]
        frame = pd.read_csv(record.csv_path)
        if record.augmentation_kind:
            # Local import avoids a data <-> augmentation import cycle.
            from .trajectory_augmentation import augment_track_frame
            if record.augmentation_kind == "partition_track_smote":
                if record.augmentation_secondary_csv_path is None:
                    raise ValueError(f"SMOTE record {record.trajectory_id} has no secondary source")
                secondary = pd.read_csv(record.augmentation_secondary_csv_path)
                frame = augment_track_frame(
                    frame, kind=record.augmentation_kind, seed=record.augmentation_seed,
                    secondary_frame=secondary, interpolation_alpha=record.augmentation_interpolation_alpha,
                )
            else:
                frame = augment_track_frame(
                    frame, kind=record.augmentation_kind, seed=record.augmentation_seed,
                    frame_drop_fraction=record.augmentation_frame_drop_fraction,
                    amplitude_scale_max_delta=record.augmentation_amplitude_scale_max_delta,
                    snr_offset_db=record.augmentation_snr_offset_db,
                )
        encoded = encode_track(
            frame,
            height_missing=record.height_missing,
            phase_missing=record.phase_missing,
        )
        return TrajectoryItem(
            trajectory_id=record.trajectory_id,
            sequence=torch.from_numpy(encoded.sequence),
            physical=torch.from_numpy(encoded.physical),
            label=record.label,
        )


@dataclass(frozen=True)
class TrajectoryBatch:
    trajectory_ids: list[str]
    sequence: Tensor
    physical: Tensor
    padding_mask: Tensor
    labels: Tensor


def collate_trajectories(items: list[TrajectoryItem]) -> TrajectoryBatch:
    if not items:
        raise ValueError("cannot collate an empty trajectory batch")
    max_length = max(item.sequence.shape[0] for item in items)
    sequence = torch.zeros((len(items), max_length, 15), dtype=torch.float32)
    padding_mask = torch.ones((len(items), max_length), dtype=torch.bool)
    physical = torch.stack([item.physical for item in items])
    labels = torch.tensor([item.label for item in items], dtype=torch.long)
    for index, item in enumerate(items):
        length = item.sequence.shape[0]
        sequence[index, :length] = item.sequence
        padding_mask[index, :length] = False
    return TrajectoryBatch(
        trajectory_ids=[item.trajectory_id for item in items],
        sequence=sequence,
        physical=physical,
        padding_mask=padding_mask,
        labels=labels,
    )

"""Persistent cache for encoded TR records, including virtual trajectories."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .data import TrajectoryDataset, TrajectoryItem, TrajectoryRecord


DEFAULT_TRAJECTORY_CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache" / "tr_encoded"


def cache_identity(records: Iterable[TrajectoryRecord], context: dict[str, object] | None = None) -> dict[str, object]:
    items = []
    for record in records:
        try:
            stat = record.csv_path.stat()
            source = {"path": str(record.csv_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            source = {"path": str(record.csv_path), "missing": True}
        secondary = None
        if record.augmentation_secondary_csv_path is not None:
            try:
                stat = record.augmentation_secondary_csv_path.stat()
                secondary = {"path": str(record.augmentation_secondary_csv_path), "size": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns}
            except OSError:
                secondary = {"path": str(record.augmentation_secondary_csv_path), "missing": True}
        items.append({"trajectory_id": record.trajectory_id, "label": record.label,
                      "augmentation_kind": record.augmentation_kind, "augmentation_seed": record.augmentation_seed,
                      "augmentation_source_trajectory_id": record.augmentation_source_trajectory_id,
                      "frame_drop_fraction": record.augmentation_frame_drop_fraction,
                      "amplitude_delta": record.augmentation_amplitude_scale_max_delta,
                      "snr_delta": record.augmentation_snr_offset_db,
                      "source_trajectory_id_b": record.augmentation_source_trajectory_id_b,
                      "interpolation_alpha": record.augmentation_interpolation_alpha,
                      "source": source, "secondary_source": secondary})
    return {"schema": 4, "augmentation_cache": "partition_local_smote_or_perturbation",
            "context": context or {}, "records": items}


class CachedTrajectoryDataset(Dataset[TrajectoryItem]):
    def __init__(self, items: list[TrajectoryItem]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TrajectoryItem:
        return self.items[index]


def load_or_build(records: list[TrajectoryRecord], cache_root: Path,
                  context: dict[str, object] | None = None) -> tuple[CachedTrajectoryDataset, dict[str, object]]:
    identity = cache_identity(records, context)
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{digest}.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("identity") == identity:
            return CachedTrajectoryDataset(payload["items"]), {"path": str(path), "key": digest, "hit": True, "records": len(records)}
    source = TrajectoryDataset(records)
    items = [source[index] for index in range(len(source))]
    temporary = path.with_suffix(".tmp")
    torch.save({"identity": identity, "items": items}, temporary)
    os.replace(temporary, path)
    return CachedTrajectoryDataset(items), {"path": str(path), "key": digest, "hit": False, "records": len(records)}

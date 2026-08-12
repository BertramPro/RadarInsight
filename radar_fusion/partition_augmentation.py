"""Deterministic, partition-local virtual trajectory augmentation.

The helpers in this module never edit a split manifest or a source file.  A
virtual record always carries the partition and the id of the source record it
was derived from, which makes diagnostics auditable and prevents accidental
cross-partition reuse.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import TrajectoryRecord
from .track_features import PHYSICAL_FEATURE_COLUMNS, encode_track

CLASS_COUNT = 5
PARTITIONS = frozenset({"train", "val", "test"})
AUGMENTATION_METHODS = frozenset({"perturbation", "smote"})
SMOTE_MAX_NEIGHBORS = 5
SMOTE_ALPHA_RANGE = (0.1, 0.9)
SMOTE_FEATURE_SPACE = "standardized_22d_physical_features"


def default_targets(records: Iterable[object]) -> list[int]:
    counts = Counter(int(record.label) for record in records)
    return [max(counts.values(), default=0)] * CLASS_COUNT


def validate_targets(values: Sequence[int] | None, records: Iterable[object]) -> list[int]:
    if values is None:
        return default_targets(records)
    if len(values) != CLASS_COUNT:
        raise ValueError("partition augmentation targets must contain exactly five class counts")
    result = [int(value) for value in values]
    if any(value < 0 or value > 10000 for value in result):
        raise ValueError("partition augmentation targets must be between 0 and 10000")
    return result


def _seed(seed: int, partition: str, label: int, source_id: str, copy_index: int) -> int:
    token = f"{int(seed)}|{partition}|{label}|{source_id}|{copy_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**31 - 1)


def _copy_plan(records: Iterable[object], partition: str, targets: Sequence[int], seed: int,
               enabled: Sequence[bool] | None = None):
    if partition not in PARTITIONS:
        raise ValueError(f"unsupported partition {partition!r}")
    base = list(records)
    by_label: dict[int, list[object]] = defaultdict(list)
    for record in base:
        by_label[int(record.label)].append(record)
    for values in by_label.values():
        values.sort(key=lambda record: str(record.trajectory_id))
    for label in range(CLASS_COUNT):
        sources = by_label.get(label, [])
        if enabled is not None and not bool(enabled[label]):
            continue
        extra = max(0, int(targets[label]) - len(sources))
        for extra_index in range(extra):
            source = sources[extra_index % len(sources)] if sources else None
            if source is None:
                continue
            source_copy = extra_index // len(sources) + 1
            virtual_id = f"{source.trajectory_id}|aug-{partition}-c{label}-{source_copy}"
            yield source, virtual_id, _seed(seed, partition, label, str(source.trajectory_id), source_copy), source_copy


def _trajectory_feature_matrix(records: Sequence[TrajectoryRecord]) -> np.ndarray:
    """Encode and standardize one class in the fixed 22-dimensional feature space."""
    vectors: list[np.ndarray] = []
    for record in records:
        path = Path(record.csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"SMOTE source trajectory is missing: {path}")
        encoded = encode_track(
            pd.read_csv(path),
            height_missing=record.height_missing,
            phase_missing=record.phase_missing,
        )
        vector = np.asarray(encoded.physical, dtype=np.float64)
        expected = (len(PHYSICAL_FEATURE_COLUMNS),)
        if vector.shape != expected:
            raise ValueError(
                f"SMOTE source {record.trajectory_id} has physical feature shape {vector.shape}, expected {expected}"
            )
        vectors.append(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))
    if not vectors:
        return np.empty((0, len(PHYSICAL_FEATURE_COLUMNS)), dtype=np.float64)
    matrix = np.stack(vectors)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (matrix - center) / scale


def _smote_copy_plan(records: Iterable[TrajectoryRecord], partition: str, targets: Sequence[int], seed: int,
                     enabled: Sequence[bool] | None = None):
    """Plan deterministic within-class k-nearest-neighbour trajectory SMOTE."""
    if partition not in PARTITIONS:
        raise ValueError(f"unsupported partition {partition!r}")
    base = sorted(list(records), key=lambda record: str(record.trajectory_id))
    by_label: dict[int, list[TrajectoryRecord]] = defaultdict(list)
    for record in base:
        by_label[int(record.label)].append(record)
    if not any(
        (enabled is None or bool(enabled[label]))
        and int(targets[label]) > len(by_label.get(label, []))
        for label in range(CLASS_COUNT)
    ):
        return
    for label in range(CLASS_COUNT):
        sources = by_label.get(label, [])
        if enabled is not None and not bool(enabled[label]):
            continue
        extra = max(0, int(targets[label]) - len(sources))
        if extra and len(sources) < 2:
            raise ValueError(f"SMOTE needs at least two {partition} trajectories for class {label}")
        if not extra:
            continue
        # Distances must be invariant to the feature distribution of other
        # classes.  Standardize within this partition/class before k-NN.
        class_features = _trajectory_feature_matrix(sources)
        feature_by_id = {
            str(record.trajectory_id): class_features[index]
            for index, record in enumerate(sources)
        }
        for extra_index in range(extra):
            primary = sources[extra_index % len(sources)]
            source_copy = extra_index // len(sources) + 1
            value_seed = _seed(seed, partition, label, str(primary.trajectory_id), source_copy)
            primary_vector = feature_by_id[str(primary.trajectory_id)]
            neighbors = sorted(
                (candidate for candidate in sources if candidate.trajectory_id != primary.trajectory_id),
                key=lambda candidate: (
                    float(np.linalg.norm(primary_vector - feature_by_id[str(candidate.trajectory_id)])),
                    str(candidate.trajectory_id),
                ),
            )
            k_neighbors = min(SMOTE_MAX_NEIGHBORS, len(neighbors))
            candidates = neighbors[:k_neighbors]
            rng = np.random.default_rng(value_seed)
            secondary = candidates[int(rng.integers(0, k_neighbors))]
            alpha = float(rng.uniform(*SMOTE_ALPHA_RANGE))
            virtual_id = f"{primary.trajectory_id}|smote-{partition}-c{label}-{source_copy}"
            yield primary, secondary, virtual_id, value_seed, source_copy, alpha, [
                str(candidate.trajectory_id) for candidate in candidates
            ]


def _method(value: str) -> str:
    result = str(value).strip().lower()
    if result not in AUGMENTATION_METHODS:
        raise ValueError(f"unsupported partition augmentation method {value!r}")
    return result


def expand_trajectory_records(
    records: Iterable[TrajectoryRecord], *, partition: str, targets: Sequence[int] | None,
    seed: int, frame_drop_fraction: float = 0.10, amplitude_scale_max_delta: float = 0.05,
    snr_offset_db: float = 1.0, allow_frame_drop: bool = True,
    enabled: Sequence[bool] | None = None,
    method: str = "perturbation",
) -> tuple[list[TrajectoryRecord], dict[str, object]]:
    """Return originals plus exactly enough deterministic virtual tracks."""
    method = _method(method)
    base = list(records)
    final_targets = validate_targets(targets, base)
    if enabled is not None and len(enabled) != CLASS_COUNT:
        raise ValueError("partition augmentation switches require exactly five values")
    result = list(base)
    manifest: list[dict[str, object]] = []
    if method == "smote":
        for source, secondary, virtual_id, value_seed, copy_index, alpha, neighbor_candidates in _smote_copy_plan(
            base, partition, final_targets, seed, enabled
        ):
            result.append(replace(
                source, trajectory_id=virtual_id, augmentation_kind="partition_track_smote",
                augmentation_seed=value_seed, augmentation_source_trajectory_id=source.trajectory_id,
                augmentation_secondary_csv_path=secondary.csv_path,
                augmentation_source_trajectory_id_b=secondary.trajectory_id,
                augmentation_interpolation_alpha=alpha,
                augmentation_frame_drop_fraction=0.0,
                augmentation_amplitude_scale_max_delta=0.0,
                augmentation_snr_offset_db=0.0,
            ))
            manifest.append({"partition": partition, "trajectory_id": virtual_id,
                             "source_trajectory_id": source.trajectory_id,
                             "source_trajectory_id_b": secondary.trajectory_id,
                             "label": int(source.label), "copy_index": copy_index,
                             "seed": value_seed, "interpolation_alpha": alpha,
                             "k_neighbors": len(neighbor_candidates),
                             "neighbor_candidates": neighbor_candidates,
                             "feature_space": SMOTE_FEATURE_SPACE})
    else:
        for source, virtual_id, value_seed, copy_index in _copy_plan(base, partition, final_targets, seed, enabled):
            result.append(replace(
                source, trajectory_id=virtual_id, augmentation_kind="partition_track_t1",
                augmentation_seed=value_seed, augmentation_source_trajectory_id=source.trajectory_id,
                augmentation_frame_drop_fraction=float(frame_drop_fraction if allow_frame_drop else 0.0),
                augmentation_amplitude_scale_max_delta=float(amplitude_scale_max_delta),
                augmentation_snr_offset_db=float(snr_offset_db),
            ))
            manifest.append({"partition": partition, "trajectory_id": virtual_id,
                             "source_trajectory_id": source.trajectory_id, "label": int(source.label),
                             "copy_index": copy_index, "seed": value_seed,
                             "frame_drop_fraction": float(frame_drop_fraction if allow_frame_drop else 0.0),
                             "amplitude_scale_max_delta": float(amplitude_scale_max_delta),
                             "snr_offset_db": float(snr_offset_db)})
    base_counts = [sum(record.label == label for record in base) for label in range(CLASS_COUNT)]
    expanded_counts = [sum(record.label == label for record in result) for label in range(CLASS_COUNT)]
    cache_parameters = {
        "schema": 1,
        "partition": partition,
        "method": method,
        "targets": final_targets,
        "seed": int(seed),
        "augmentation_enabled": [True] * CLASS_COUNT if enabled is None else [bool(value) for value in enabled],
        "allow_frame_drop": bool(allow_frame_drop),
        "frame_drop_fraction": float(frame_drop_fraction),
        "amplitude_scale_max_delta": float(amplitude_scale_max_delta),
        "snr_offset_db": float(snr_offset_db),
    }
    return result, {"partition": partition, "method": method, "targets": final_targets,
                    "seed": int(seed), "cache_parameters": cache_parameters,
                    "augmentation_enabled": [True] * CLASS_COUNT if enabled is None else [bool(value) for value in enabled],
                    "base_counts": base_counts,
                    "expanded_counts": expanded_counts, "virtual_count": len(manifest), "records": manifest,
                    "smote_strategy": ({"feature_space": SMOTE_FEATURE_SPACE,
                                        "max_neighbors": SMOTE_MAX_NEIGHBORS,
                                        "alpha_range": list(SMOTE_ALPHA_RANGE)} if method == "smote" else None),
                    "kind": "partition_local_related_virtual_trajectories"}


def expand_rd_frames(frames, *, partition: str, targets: Sequence[int] | None, seed: int,
                     method: str = "perturbation", smote_plan: dict[str, object] | None = None):
    """Expand complete RD trajectories; every virtual track keeps all source frames."""
    method = _method(method)
    base = list(frames)
    table: dict[str, object] = {}
    for frame in base:
        table.setdefault(str(frame.trajectory_id), frame)
    # ``targets`` is the requested five-class final count.  The previous
    # argument order accidentally treated the source-frame table as targets,
    # which only surfaced once RD virtual tracks were actually requested.
    # Validate against one representative frame per trajectory, because the
    # targets are trajectory counts rather than frame counts.
    final_targets = validate_targets(targets, table.values())
    cache_parameters = {
        "schema": 1,
        "partition": partition,
        "method": method,
        "targets": final_targets,
        "seed": int(seed),
        "rd_augmentation": {
            "intensity_offset": 1.5,
            "noise_std": 0.35,
            "range_shift_pixels": 1,
            "velocity_shift_pixels": 3,
        },
    }
    cache_plan_key = hashlib.sha256(
        json.dumps(cache_parameters, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result = list(base)
    manifest: list[dict[str, object]] = []
    if method == "smote":
        if not isinstance(smote_plan, dict) or smote_plan.get("method") != "smote":
            raise ValueError("RD SMOTE requires the matching TR SMOTE plan")
        if smote_plan.get("partition") != partition:
            raise ValueError("RD SMOTE plan belongs to a different partition")
        if list(smote_plan.get("targets") or []) != final_targets:
            raise ValueError("RD SMOTE plan targets do not match the requested RD targets")
        by_trajectory: dict[str, list[object]] = defaultdict(list)
        for frame in base:
            by_trajectory[str(frame.trajectory_id)].append(frame)
        for values in by_trajectory.values():
            values.sort(key=lambda frame: str(frame.path))
        for plan_record in smote_plan.get("records") or []:
            source_id = str(plan_record["source_trajectory_id"])
            secondary_id = str(plan_record["source_trajectory_id_b"])
            virtual_id = str(plan_record["trajectory_id"])
            value_seed = int(plan_record["seed"])
            copy_index = int(plan_record["copy_index"])
            alpha = float(plan_record["interpolation_alpha"])
            if source_id == secondary_id:
                raise ValueError(f"RD SMOTE plan pairs trajectory {source_id} with itself")
            if source_id not in table or secondary_id not in table:
                raise ValueError(f"RD SMOTE source is missing for virtual trajectory {virtual_id}")
            source, secondary = table[source_id], table[secondary_id]
            if int(source.label) != int(secondary.label):
                raise ValueError(f"RD SMOTE sources have different labels for {virtual_id}")
            primary_frames, secondary_frames = by_trajectory[source_id], by_trajectory[secondary_id]
            for index, frame in enumerate(primary_frames):
                secondary_index = int(round(index * max(len(secondary_frames) - 1, 0) / max(len(primary_frames) - 1, 1)))
                companion = secondary_frames[secondary_index]
                result.append(replace(frame, trajectory_id=virtual_id, augmentation_kind="partition_rd_smote",
                                      augmentation_seed=value_seed, augmentation_source_trajectory_id=source_id,
                                      augmentation_secondary_path=str(companion.path),
                                      augmentation_source_trajectory_id_b=secondary_id,
                                      augmentation_interpolation_alpha=alpha,
                                      augmentation_cache_plan_key=cache_plan_key))
            manifest.append({"partition": partition, "trajectory_id": virtual_id,
                             "source_trajectory_id": source_id, "source_trajectory_id_b": secondary_id,
                             "label": int(source.label), "copy_index": copy_index, "seed": value_seed,
                             "interpolation_alpha": alpha,
                             "k_neighbors": int(plan_record.get("k_neighbors", 0)),
                             "neighbor_candidates": list(plan_record.get("neighbor_candidates") or []),
                             "feature_space": plan_record.get("feature_space", SMOTE_FEATURE_SPACE),
                             "frame_count": len(primary_frames)})
    else:
        for source, virtual_id, value_seed, copy_index in _copy_plan(table.values(), partition, final_targets, seed):
            source_id = str(source.trajectory_id)
            for frame in (item for item in base if str(item.trajectory_id) == source_id):
                result.append(replace(frame, trajectory_id=virtual_id, augmentation_kind="partition_rd_t1",
                                      augmentation_seed=value_seed,
                                      augmentation_source_trajectory_id=source_id,
                                      augmentation_cache_plan_key=cache_plan_key))
            manifest.append({"partition": partition, "trajectory_id": virtual_id,
                             "source_trajectory_id": source_id, "label": int(source.label),
                             "copy_index": copy_index, "seed": value_seed})
    base_counts = [sum(item.label == label for item in table.values()) for label in range(CLASS_COUNT)]
    expanded_table = {str(frame.trajectory_id): frame for frame in result}
    expanded_counts = [sum(item.label == label for item in expanded_table.values()) for label in range(CLASS_COUNT)]
    return result, {"partition": partition, "method": method, "targets": final_targets, "seed": int(seed),
                    "cache_parameters": cache_parameters, "cache_plan_key": cache_plan_key, "base_counts": base_counts,
                    "expanded_counts": expanded_counts, "virtual_count": len(manifest), "records": manifest,
                    "smote_strategy": (smote_plan.get("smote_strategy") if method == "smote" and smote_plan else None),
                    "kind": "partition_local_related_virtual_trajectories"}

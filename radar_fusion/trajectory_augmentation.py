"""Train-only virtual trajectory augmentation.

This module deliberately works on an in-memory DataFrame.  It never writes a
source CSV or changes the grouped split, so train-only augmentation cannot
leak a copied trajectory into validation or test.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from .data import TrajectoryRecord


SUPPORTED_KINDS = frozenset({"unknown_track_t1", "bird_track_t1", "partition_track_t1", "partition_track_smote"})


def _resample_numeric(values: pd.Series, length: int) -> np.ndarray:
    """Linearly resample a numeric trajectory column on normalized time."""
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    valid = np.isfinite(raw)
    if not valid.any():
        return np.full(length, np.nan, dtype=np.float64)
    source_x = np.linspace(0.0, 1.0, num=len(raw), dtype=np.float64)[valid]
    target_x = np.linspace(0.0, 1.0, num=length, dtype=np.float64)
    return np.interp(target_x, source_x, raw[valid])


def smote_track_frame(frame: pd.DataFrame, secondary_frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Create one same-class trajectory by interpolating two time-normalized tracks.

    This is sequence-aware SMOTE: both source tracks are resampled to a common
    normalized time axis before their numeric observation columns are mixed.
    The resulting frame is subsequently encoded by the normal B01 pipeline,
    so its 15-dimensional sequence and 22 physical features stay consistent.
    """
    if frame.empty or secondary_frame.empty:
        raise ValueError("SMOTE requires two non-empty source trajectories")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("SMOTE interpolation alpha must be in (0, 1)")
    length = max(3, max(len(frame), len(secondary_frame)))
    result = pd.DataFrame(index=range(length))
    columns = list(dict.fromkeys([*frame.columns, *secondary_frame.columns]))
    for column in columns:
        left = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        right = secondary_frame[column] if column in secondary_frame else pd.Series(np.nan, index=secondary_frame.index)
        left_numeric = pd.to_numeric(left, errors="coerce")
        right_numeric = pd.to_numeric(right, errors="coerce")
        if left_numeric.notna().any() and right_numeric.notna().any():
            result[column] = (1.0 - alpha) * _resample_numeric(left, length) + alpha * _resample_numeric(right, length)
        else:
            source = left if left.notna().any() else right
            source_index = np.linspace(0, max(len(source) - 1, 0), num=length).round().astype(int)
            result[column] = source.iloc[source_index].to_numpy() if len(source) else np.nan
    return result


def augment_track_frame(
    frame: pd.DataFrame,
    *,
    kind: str,
    seed: int,
    frame_drop_fraction: float = 0.10,
    amplitude_scale_max_delta: float = 0.05,
    snr_offset_db: float = 1.0,
    secondary_frame: pd.DataFrame | None = None,
    interpolation_alpha: float = 0.5,
) -> pd.DataFrame:
    """Return one deterministic, physically coherent virtual track copy."""
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported trajectory augmentation kind: {kind!r}")
    if kind == "partition_track_smote":
        if secondary_frame is None:
            raise ValueError("partition_track_smote requires a secondary trajectory")
        return smote_track_frame(frame, secondary_frame, interpolation_alpha)
    result = frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    fraction = min(0.40, max(0.0, float(frame_drop_fraction)))
    drop_count = min(int(round(len(result) * fraction)), max(0, len(result) - 3))
    if drop_count:
        start = int(rng.integers(0, len(result) - drop_count + 1))
        result = result.drop(index=range(start, start + drop_count)).reset_index(drop=True)
    amplitude_delta = max(0.0, float(amplitude_scale_max_delta))
    if "amplitude_db" in result and amplitude_delta:
        values = pd.to_numeric(result["amplitude_db"], errors="coerce").to_numpy(dtype=np.float64)
        scale = 1.0 + float(rng.uniform(-amplitude_delta, amplitude_delta))
        result["amplitude_db"] = np.where(np.isfinite(values), values * scale, values)
    snr_delta = max(0.0, float(snr_offset_db))
    if "snr_db" in result and snr_delta:
        values = pd.to_numeric(result["snr_db"], errors="coerce").to_numpy(dtype=np.float64)
        offset = float(rng.uniform(-snr_delta, snr_delta))
        result["snr_db"] = np.where(np.isfinite(values), values + offset, values)
    return result


def append_training_copies(
    records: Iterable[TrajectoryRecord],
    *,
    label: int,
    copies: int = 0,
    kind: str,
    seed: int,
    frame_drop_fraction: float = 0.10,
    amplitude_scale_max_delta: float = 0.05,
    snr_offset_db: float = 1.0,
) -> list[TrajectoryRecord]:
    """Append deterministic virtual records for one class.

    ``copies`` is the exact number of virtual records to append.  This is
    intentionally different from the old project's misleading
    ``*_target_count`` spelling: it is *not* the desired final class count.
    """
    base = list(records)
    sources = [record for record in base if record.label == label]
    if not sources:
        return base
    extra = max(0, int(copies))
    if not extra:
        return base
    result = list(base)
    for extra_index in range(extra):
        source_index = extra_index % len(sources)
        source = sources[source_index]
        copy_index = extra_index // len(sources) + 1
        result.append(replace(
            source,
            trajectory_id=f"{source.trajectory_id}|aug-{kind}-{copy_index}",
            augmentation_kind=kind,
            augmentation_seed=int(seed) + 10007 + label * 1000 + source_index * 131 + extra_index,
            augmentation_source_trajectory_id=source.trajectory_id,
            augmentation_frame_drop_fraction=float(frame_drop_fraction),
            augmentation_amplitude_scale_max_delta=float(amplitude_scale_max_delta),
            augmentation_snr_offset_db=float(snr_offset_db),
        ))
    return result


def expand_training_records(
    records: Iterable[TrajectoryRecord],
    *,
    label: int,
    target_count: int = 0,
    copies: int = 0,
    kind: str,
    seed: int,
    frame_drop_fraction: float = 0.10,
    amplitude_scale_max_delta: float = 0.05,
    snr_offset_db: float = 1.0,
) -> list[TrajectoryRecord]:
    """Compatibility wrapper for old callers using the legacy option name."""
    if copies and target_count:
        raise ValueError("specify copies or legacy target_count, not both")
    return append_training_copies(
        records, label=label, copies=copies or target_count, kind=kind, seed=seed,
        frame_drop_fraction=frame_drop_fraction,
        amplitude_scale_max_delta=amplitude_scale_max_delta, snr_offset_db=snr_offset_db,
    )


def label_counts(records: Iterable[TrajectoryRecord]) -> dict[int, int]:
    return dict(Counter(record.label for record in records))

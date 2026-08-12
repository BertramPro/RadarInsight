"""CQ-08 trajectory features migrated from K:\\radar\\main.

The definitions intentionally match the B01 checkpoint contract. Changing a
column, transform, or statistic here invalidates direct B01 weight reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRACK_FEATURE_COLUMNS = [
    "time_seconds",
    "azimuth_deg",
    "range_m",
    "radial_speed_mps",
    "elevation_deg",
    "height_m",
    "amplitude_db",
    "snr_db",
    "course_deg",
    "speed_mps",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "tangential_ratio",
    "nd_count",
]

PHYSICAL_FEATURE_COLUMNS = [
    "height_mean",
    "height_std",
    "height_slope",
    "ground_speed_mean",
    "ground_speed_std",
    "ground_speed_max",
    "speed_volatility",
    "speed_trend",
    "course_change_rate_mean",
    "trajectory_curvature_mean",
    "turn_agility",
    "acceleration_mean",
    "acceleration_std",
    "oscillation_factor",
    "hover_ratio",
    "radial_speed_std",
    "radial_speed_flatness",
    "amplitude_mean",
    "amplitude_std",
    "snr_mean",
    "height_missing_flag",
    "phase_missing_flag",
]


@dataclass(frozen=True)
class EncodedTrack:
    sequence: np.ndarray
    physical: np.ndarray


def _safe_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _linear_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    x_centered = x - float(x.mean())
    denominator = float(np.square(x_centered).sum())
    if denominator <= 0.0:
        return 0.0
    return float((x_centered * (values - float(values.mean()))).sum()) / denominator


def _angle_diff(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    difference = np.diff(values, prepend=values[:1])
    return (difference + 180.0) % 360.0 - 180.0


def _spectral_flatness(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    magnitude = np.abs(values.astype(np.float64)) + 1e-6
    arithmetic_mean = float(magnitude.mean())
    if arithmetic_mean <= 0.0:
        return 0.0
    return float(np.exp(np.log(magnitude).mean())) / arithmetic_mean


def encode_track_sequence(frame: pd.DataFrame) -> np.ndarray:
    missing = [column for column in TRACK_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"trajectory CSV lacks B01 columns: {missing}")
    cleaned = frame[TRACK_FEATURE_COLUMNS].copy().replace([np.inf, -np.inf], np.nan)
    for column in TRACK_FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["nd_count"] = cleaned["nd_count"].fillna(-1.0)
    return cleaned.fillna(0.0).to_numpy(dtype=np.float32)


def extract_physical_features(
    frame: pd.DataFrame,
    *,
    height_missing: bool = True,
    phase_missing: bool = True,
) -> np.ndarray:
    """Return the exact 22-feature B01 physical vector."""
    if frame.empty:
        return np.zeros(len(PHYSICAL_FEATURE_COLUMNS), dtype=np.float32)

    eps = 1e-6
    height = _safe_numeric(frame["height_m"])
    speed = _safe_numeric(frame["speed_mps"])
    course = _safe_numeric(frame["course_deg"])
    radial_speed = _safe_numeric(frame["radial_speed_mps"])
    amplitude = _safe_numeric(frame["amplitude_db"])
    snr = _safe_numeric(frame["snr_db"])
    azimuth = _safe_numeric(frame["azimuth_deg"])
    track_range = _safe_numeric(frame["range_m"])

    height_mean = float(height.mean()) if height.size else 0.0
    height_std = float(height.std()) if height.size else 0.0
    speed_mean = float(speed.mean()) if speed.size else 0.0
    speed_std = float(speed.std()) if speed.size else 0.0
    speed_diff = np.diff(speed, prepend=speed[:1]) if speed.size else np.zeros(0, dtype=np.float32)
    course_diff = _angle_diff(course)
    course_change_std = float(np.abs(course_diff).std()) if course_diff.size else 0.0

    oscillation_factor = 0.0
    if course.size >= 3:
        heading_delta = _angle_diff(course)[1:]
        signs = np.zeros(len(heading_delta), dtype=np.int8)
        signs[heading_delta > 0.5] = 1
        signs[heading_delta < -0.5] = -1
        weights = [1.0, 1.5, 2.0, 3.0, 5.0]
        oscillation_count = 0
        last_index = -2
        for index in range(len(signs)):
            adjacent = (
                index >= 1
                and signs[index - 1] != 0
                and signs[index] != 0
                and signs[index - 1] + signs[index] == 0
            )
            separated = (
                index >= 2
                and signs[index - 2] != 0
                and signs[index] != 0
                and signs[index - 2] + signs[index] == 0
                and signs[index - 1] == 0
            )
            if (adjacent or separated) and index - last_index > 1:
                oscillation_factor += weights[min(oscillation_count, len(weights) - 1)] * abs(heading_delta[index])
                oscillation_count += 1
                last_index = index

    azimuth_diff = _angle_diff(azimuth)
    range_diff = np.diff(track_range, prepend=track_range[:1]) if track_range.size else np.zeros(0, dtype=np.float32)
    curvature = np.sqrt(np.square(azimuth_diff) + np.square(range_diff))
    amplitude_mean = float(np.log1p(float(amplitude.mean()))) if amplitude.size else 0.0
    amplitude_std = float(np.log1p(float(amplitude.std()))) if amplitude.size else 0.0

    values = np.array(
        [
            height_mean,
            height_std,
            _linear_slope(height),
            speed_mean,
            speed_std,
            float(speed.max()) if speed.size else 0.0,
            speed_std / (speed_mean + eps) if speed.size else 0.0,
            _linear_slope(speed),
            float(np.abs(course_diff).mean()) if course_diff.size else 0.0,
            float(curvature.mean()) if curvature.size else 0.0,
            float(np.log1p(course_change_std / (speed_mean + eps))),
            float(speed_diff.mean()) if speed_diff.size else 0.0,
            float(speed_diff.std()) if speed_diff.size else 0.0,
            float(np.log1p(oscillation_factor)),
            float((np.abs(speed) <= 1.0).mean()) if speed.size else 0.0,
            float(radial_speed.std()) if radial_speed.size else 0.0,
            _spectral_flatness(radial_speed),
            amplitude_mean,
            amplitude_std,
            float(snr.mean()) if snr.size else 0.0,
            float(height_missing),
            float(phase_missing),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def encode_track(frame: pd.DataFrame, *, height_missing: bool = True, phase_missing: bool = True) -> EncodedTrack:
    return EncodedTrack(
        sequence=encode_track_sequence(frame),
        physical=extract_physical_features(
            frame,
            height_missing=height_missing,
            phase_missing=phase_missing,
        ),
    )

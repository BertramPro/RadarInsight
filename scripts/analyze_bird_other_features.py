"""Extract physical RD features for five-class and Bird/Other groups."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "radar_rd"))
from train import RDCache, SmallRDCNN, build_manifest, split_frames  # noqa: E402
from analyze_rd_attribution import frame_predictions  # noqa: E402


CONFUSION_GROUPS = {
    "bird_to_other": (1, 4),
    "other_to_bird": (4, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_features(frame, config: dict[str, object], rd_cache: RDCache) -> dict[str, object]:
    rd, _ = rd_cache.load(frame.path)
    rd = np.clip(np.asarray(rd, dtype=np.float32), 0.0, 100.0)
    rows, width = rd.shape
    velocity = np.linspace(float(config.get("velocity_min", -90.0)),
                           float(config.get("velocity_max", 89.0)), width)
    relative_power = np.power(10.0, (rd - float(rd.max())) / 10.0)
    central = np.abs(velocity) <= 10.0
    near_zero_energy_ratio = float(relative_power[:, central].sum() / max(float(relative_power.sum()), 1e-12))

    profile = rd.max(axis=0)
    baseline = float(np.percentile(profile, 20.0))
    prominence = max(float(profile.max()) - baseline, 1e-6)
    threshold = baseline + 0.5 * prominence
    peak = int(np.argmax(profile))
    active = profile >= threshold
    left, right = peak, peak
    while left > 0 and active[left - 1]:
        left -= 1
    while right + 1 < width and active[right + 1]:
        right += 1
    step = float(abs(velocity[1] - velocity[0])) if width > 1 else 0.0
    response_width_mps = float((right - left + 1) * step)

    window = max(5, int(round(width * 0.04)))
    if window % 2 == 0:
        window += 1
    padding = window // 2
    padded = np.pad(rd, ((0, 0), (padding, padding)), mode="edge")
    prefix = np.cumsum(np.pad(padded, ((0, 0), (1, 0)), mode="constant"), axis=1)
    local_mean = (prefix[:, window:] - prefix[:, :-window]) / float(window)
    local_contrast = (rd - local_mean) / max(float(config.get("normalization_std", 1.0)), 1e-6)
    positive_contrast = np.maximum(local_contrast, 0.0)
    local_contrast_peak = float(positive_contrast.max())
    top_count = max(1, int(positive_contrast.size * 0.05))
    local_contrast_top5_mean = float(np.sort(positive_contrast.reshape(-1))[-top_count:].mean())

    profile_power = relative_power.mean(axis=0)
    profile_power = profile_power / max(float(profile_power.sum()), 1e-12)
    range_profile = relative_power.mean(axis=1)
    range_profile = range_profile / max(float(range_profile.sum()), 1e-12)
    range_peak = int(np.argmax(range_profile))
    range_baseline = float(np.percentile(range_profile, 20.0))
    range_threshold = range_baseline + 0.5 * max(float(range_profile.max()) - range_baseline, 1e-12)
    range_active = range_profile >= range_threshold
    range_left, range_right = range_peak, range_peak
    while range_left > 0 and range_active[range_left - 1]:
        range_left -= 1
    while range_right + 1 < rows and range_active[range_right + 1]:
        range_right += 1
    return {
        "near_zero_energy_ratio": near_zero_energy_ratio,
        "response_width_mps": response_width_mps,
        "local_contrast_peak": local_contrast_peak,
        "local_contrast_top5_mean": local_contrast_top5_mean,
        "peak_velocity_mps": float(velocity[peak]),
        "range_peak_row": range_peak,
        "range_response_width_cells": int(range_right - range_left + 1),
        "range_peak_energy_ratio": float(range_profile[range_peak]),
        "profile": profile_power.tolist(),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def trajectory_features(frames, config, rd_cache) -> dict[str, object]:
    values = [frame_features(frame, config, rd_cache) for frame in frames]
    profiles = np.asarray([item["profile"] for item in values], dtype=np.float64)
    mean_profile = profiles.mean(axis=0)
    widths = np.asarray([item["response_width_mps"] for item in values], dtype=np.float64)
    peaks = np.asarray([item["peak_velocity_mps"] for item in values], dtype=np.float64)
    near_zero = np.asarray([item["near_zero_energy_ratio"] for item in values], dtype=np.float64)
    contrast_peak = np.asarray([item["local_contrast_peak"] for item in values], dtype=np.float64)
    contrast_top = np.asarray([item["local_contrast_top5_mean"] for item in values], dtype=np.float64)
    range_peaks = np.asarray([item["range_peak_row"] for item in values], dtype=np.float64)
    range_widths = np.asarray([item["range_response_width_cells"] for item in values], dtype=np.float64)
    range_concentration = np.asarray([item["range_peak_energy_ratio"] for item in values], dtype=np.float64)
    return {
        "frame_count": len(values),
        "near_zero_energy_ratio": float(near_zero.mean()),
        "near_zero_energy_ratio_std": float(near_zero.std()),
        "response_width_mps": float(widths.mean()),
        "response_width_std_mps": float(widths.std()),
        "response_width_cv": float(widths.std() / max(abs(widths.mean()), 1e-6)),
        "local_contrast_peak": float(contrast_peak.mean()),
        "local_contrast_top5_mean": float(contrast_top.mean()),
        "peak_velocity_mean_mps": float(peaks.mean()),
        "peak_velocity_std_mps": float(peaks.std()),
        "range_peak_row_mean": float(range_peaks.mean()),
        "range_peak_row_std": float(range_peaks.std()),
        "range_response_width_cells": float(range_widths.mean()),
        "range_peak_energy_ratio": float(range_concentration.mean()),
        "profile_cosine_to_mean": float(np.mean([cosine(profile, mean_profile) for profile in profiles])),
        "adjacent_profile_cosine": float(np.mean([cosine(profiles[i - 1], profiles[i]) for i in range(1, len(profiles))])) if len(profiles) > 1 else 1.0,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "frame_count"]
    result = {"trajectory_count": len(rows), "frame_count": int(sum(int(row["frame_count"]) for row in rows))}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {"mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std())}
    return result


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_json(checkpoint.parent / "config.json")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    input_channels = 1 if config.get("input_mode", "rd") == "rd" else 2
    model = SmallRDCNN(input_channels=input_channels, head=str(config.get("model_head", "global")))
    model.load_state_dict(checkpoint_data["model_state"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model.to(device).eval()

    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else Path(str(config["dataset_root"]))
    frames = build_manifest(dataset_root)
    split = load_json(checkpoint.parent / "split.json")
    val_frames = split_frames(frames, split)["val"]
    rd_cache = RDCache(Path(str(config["rd_cache"])), val_frames,
                       velocity_min=float(config.get("velocity_min", -90.0)),
                       velocity_max=float(config.get("velocity_max", 89.0)),
                       target_width=int(config.get("target_width", 900)),
                       resampling=str(config.get("resampling", "db_linear")))
    probabilities = frame_predictions(model, val_frames, config, args.batch_size, rd_cache, device)
    grouped = defaultdict(list)
    labels = {}
    for frame in val_frames:
        grouped[frame.trajectory_id].append(frame)
        labels[frame.trajectory_id] = frame.label
    predictions = {}
    for trajectory_id, trajectory_frames in grouped.items():
        values = np.asarray([probabilities[frame.path] for frame in trajectory_frames], dtype=np.float64)
        predictions[trajectory_id] = int(values.mean(axis=0).argmax())

    rows = []
    groups = defaultdict(list)
    for trajectory_id, trajectory_frames in grouped.items():
        pair = (labels[trajectory_id], predictions[trajectory_id])
        features = trajectory_features(trajectory_frames, config, rd_cache)
        row = {"trajectory_id": trajectory_id, "true": labels[trajectory_id], "pred": predictions[trajectory_id],
               "group": f"{['drone', 'bird', 'balloon', 'clutter', 'other'][labels[trajectory_id]]}_all", **features}
        rows.append(row)
        groups[row["group"]].append(row)
        for group_name, group_pair in CONFUSION_GROUPS.items():
            if group_pair == pair:
                groups[group_name].append({**row, "group": group_name})
    summary = {"checkpoint": str(checkpoint), "device": str(device), "input_mode": config.get("input_mode", "rd"),
               "validation_trajectory_count": len(grouped), "groups": {name: summarize(group_rows)
                                                                          for name, group_rows in groups.items()},
               "features": ["near_zero_energy_ratio", "response_width_mps", "local_contrast_peak",
                            "local_contrast_top5_mean", "profile_cosine_to_mean", "adjacent_profile_cosine",
                            "peak_velocity_std_mps", "response_width_cv", "range_peak_row_mean",
                            "range_peak_row_std", "range_response_width_cells", "range_peak_energy_ratio"]}
    (output / "trajectory_features.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

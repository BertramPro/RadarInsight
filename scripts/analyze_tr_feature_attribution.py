"""Permutation attribution for the B01-compatible trajectory branch.

The script never trains or updates model parameters.  It measures the drop in
fixed-F validation performance after one input field is independently
permuted, so the reported values describe model sensitivity rather than a
causal property of a target class.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_fusion.data import (  # noqa: E402
    TrajectoryDataset,
    collate_trajectories,
    load_grouped_split,
    load_trajectory_records,
)
from radar_fusion.model import CLASS_NAMES, load_b01_trajectory_branch  # noqa: E402
from radar_fusion.track_features import PHYSICAL_FEATURE_COLUMNS, TRACK_FEATURE_COLUMNS  # noqa: E402
from radar_fusion.trajectory_cache import DEFAULT_TRAJECTORY_CACHE_ROOT, load_or_build  # noqa: E402


DEFAULT_SPLIT = Path(r"K:\radar\main\data\manifests\cq08_grouped_split_f.json")
DEFAULT_TRACK_INDEX = Path(r"K:\radar\main\data\processed\expert1_track_index.csv")
DEFAULT_CHECKPOINT = Path(
    r"K:\radar\main\artifacts\f_protocol\20260728-183147\b01_transformer_seed42\best_model.pt"
)

# CQ-08 height is missing and these slots are structural placeholders only.
EXCLUDED_SEQUENCE_FIELDS = {"height_m"}
EXCLUDED_PHYSICAL_FIELDS = {
    "height_mean", "height_std", "height_slope", "height_missing_flag", "phase_missing_flag",
}

DISPLAY_NAMES = {
    "time_seconds": "相对时间", "azimuth_deg": "方位角", "range_m": "距离",
    "radial_speed_mps": "径向速度", "elevation_deg": "俯仰角", "amplitude_db": "回波幅度",
    "snr_db": "信噪比", "course_deg": "航向", "speed_mps": "总速度",
    "vx_mps": "x 轴速度", "vy_mps": "y 轴速度", "vz_mps": "z 轴速度",
    "tangential_ratio": "切向/径向速度比", "nd_count": "原始点数量",
    "ground_speed_mean": "平均总速度", "ground_speed_std": "总速度标准差",
    "ground_speed_max": "最大总速度", "speed_volatility": "速度波动率",
    "speed_trend": "速度趋势", "course_change_rate_mean": "平均航向变化量",
    "trajectory_curvature_mean": "离散轨迹曲折度", "turn_agility": "转向敏捷性",
    "acceleration_mean": "速度变化均值", "acceleration_std": "速度变化标准差",
    "oscillation_factor": "航向振荡因子", "hover_ratio": "低速比例",
    "radial_speed_std": "径向速度标准差", "radial_speed_flatness": "径向速度平坦度",
    "amplitude_mean": "幅度均值", "amplitude_std": "幅度标准差", "snr_mean": "平均信噪比",
}
CLASS_DISPLAY_NAMES = {
    "drone": "无人机", "bird": "飞鸟", "balloon": "空飘球",
    "clutter": "杂波", "other": "未知目标",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--track-index", type=Path, default=DEFAULT_TRACK_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def classification_metrics(records, prediction_key):
    """Small dependency-free equivalent of the project's reporting metrics."""
    truth = np.asarray([int(record["true_class"]) for record in records], dtype=np.int64)
    prediction = np.asarray([int(record[prediction_key]) for record in records], dtype=np.int64)
    report = {}
    f1_values = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(np.sum((truth == class_index) & (prediction == class_index)))
        false_positive = int(np.sum((truth != class_index) & (prediction == class_index)))
        false_negative = int(np.sum((truth == class_index) & (prediction != class_index)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        report[class_name] = {"precision": precision, "recall": recall, "f1-score": f1,
                              "support": int(np.sum(truth == class_index))}
    return {
        "trajectory_count": int(len(records)),
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(np.mean(f1_values)),
        "classification_report": report,
    }


@torch.inference_mode()
def infer(model, sequence, physical, padding_mask, labels, ids, device, batch_size):
    output = []
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        logits = model(
            sequence[start:end].to(device, non_blocking=True),
            physical[start:end].to(device, non_blocking=True),
            padding_mask[start:end].to(device, non_blocking=True),
        ).cpu()
        predictions = logits.argmax(dim=1).tolist()
        for trajectory_id, label, prediction in zip(ids[start:end], labels[start:end].tolist(), predictions):
            output.append({
                "trajectory_id": trajectory_id,
                "true_class": int(label),
                "true_label": CLASS_NAMES[int(label)],
                "prediction": int(prediction),
                "prediction_label": CLASS_NAMES[int(prediction)],
            })
    return output


def macro_f1(model, sequence, physical, padding_mask, labels, ids, device, batch_size):
    decisions = infer(model, sequence, physical, padding_mask, labels, ids, device, batch_size)
    return classification_metrics(decisions, "prediction")


def collect_tensors(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_trajectories)
    batch = next(iter(loader))
    return batch.sequence, batch.physical, batch.padding_mask, batch.labels, batch.trajectory_ids


def permutation_rows(baseline, field, source, kind, model, sequence, physical, mask, labels, ids, args):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1009 * (field + 1) + (0 if kind == "sequence" else 1))
    if kind == "sequence":
        modified_sequence = sequence.clone()
        valid = ~mask
        values = modified_sequence[:, :, field][valid]
        modified_sequence[:, :, field][valid] = values[torch.randperm(values.numel(), generator=generator)]
        metrics = macro_f1(model, modified_sequence, physical, mask, labels, ids, args.device, args.batch_size)
    else:
        modified_physical = physical.clone()
        modified_physical[:, field] = modified_physical[torch.randperm(len(ids), generator=generator), field]
        metrics = macro_f1(model, sequence, modified_physical, mask, labels, ids, args.device, args.batch_size)
    return {
        "branch": "逐点序列" if kind == "sequence" else "航迹统计",
        "field": source,
        "name": DISPLAY_NAMES[source],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "macro_f1_drop": baseline["macro_f1"] - metrics["macro_f1"],
        "accuracy_drop": baseline["accuracy"] - metrics["accuracy"],
        "class_f1": {label: metrics["classification_report"][label]["f1-score"] for label in CLASS_NAMES},
    }


def make_figure(rows, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    groups = [
        ("逐点有效观测（14 维）", [row for row in rows if row["branch"] == "逐点序列"]),
        ("航迹统计特征（17 维）", [row for row in rows if row["branch"] == "航迹统计"]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for axis, (title, group) in zip(axes, groups):
        group = sorted(group, key=lambda item: item["macro_f1_drop"])
        values = [100.0 * row["macro_f1_drop"] for row in group]
        labels = [row["name"] for row in group]
        colors = ["#C65D3B" if value > 0 else "#8EA4B8" for value in values]
        axis.barh(np.arange(len(group)), values, color=colors, height=0.68)
        axis.axvline(0, color="#4A5560", linewidth=0.8)
        axis.set_yticks(np.arange(len(group)), labels, fontsize=10)
        axis.set_xlabel("置换后的验证 Macro-F1 下降（百分点）", fontsize=11)
        axis.set_title(title, fontsize=13, pad=10)
        axis.grid(axis="x", alpha=0.22, linewidth=0.6)
    figure.suptitle("B01 航迹分支输入置换归因（固定 F 验证集，n=232）", fontsize=15)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_trajectory_figure(records, output: Path) -> None:
    """Plot class prototypes that are representative yet structurally separated."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    class_candidates = {}
    all_values = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        candidates = []
        for record in records:
            if record.label != class_index:
                continue
            frame = pd.read_csv(record.csv_path)
            if len(frame) < 2:
                continue
            speed = pd.to_numeric(frame["speed_mps"], errors="coerce").fillna(0).to_numpy(float)
            radial = pd.to_numeric(frame["radial_speed_mps"], errors="coerce").fillna(0).to_numpy(float)
            course = pd.to_numeric(frame["course_deg"], errors="coerce").fillna(0).to_numpy(float)
            course_delta = (np.diff(course) + 180.0) % 360.0 - 180.0
            signs = np.sign(course_delta[np.abs(course_delta) > 0.5])
            oscillation = float(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0.0
            feature = np.array([len(frame), speed.mean(), speed.std(), radial.std(), np.abs(course_delta).mean(), oscillation])
            candidates.append((record, feature))
            all_values.append(feature)
        class_candidates[class_name] = candidates
    if not all_values:
        return []
    all_values = np.stack(all_values)
    scale = np.median(np.abs(all_values - np.median(all_values, axis=0)), axis=0)
    scale = np.maximum(scale, 1e-3)
    centers = {
        name: np.median(np.stack([item[1] for item in candidates]), axis=0)
        for name, candidates in class_candidates.items() if candidates
    }
    chosen = []
    chosen_manifest = []
    for class_name, candidates in class_candidates.items():
        if not candidates:
            continue
        # Avoid very short tracks whose polyline cannot show a motion pattern.
        usable = [item for item in candidates if len(pd.read_csv(item[0].csv_path)) >= 8] or candidates
        center = centers[class_name]
        others = [value for name, value in centers.items() if name != class_name]
        scored = []
        for record, feature in usable:
            own_distance = float(np.linalg.norm((feature - center) / scale))
            other_distance = min(float(np.linalg.norm((feature - value) / scale)) for value in others) if others else 0.0
            # Keep the sample close to its own class structure; separation is
            # only a tie-breaker for choosing a visually informative example.
            scored.append((other_distance - 0.90 * own_distance, record, feature, own_distance, other_distance))
        _, selected, selected_feature, own_distance, other_distance = max(scored, key=lambda item: item[0])
        display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name)
        chosen.append((display_name, selected))
        chosen_manifest.append({"class": display_name, "trajectory_id": selected.trajectory_id,
                                "features": selected_feature.tolist(), "distance_to_class_center": own_distance,
                                "distance_to_nearest_other_center": other_distance})

    figure = plt.figure(figsize=(10.0, 6.3), constrained_layout=True)
    grid = figure.add_gridspec(2, 6)
    axes = [
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 0:3]),
        figure.add_subplot(grid[1, 3:6]),
    ]
    for row_index, (class_name, record) in enumerate(chosen):
        frame = pd.read_csv(record.csv_path)
        azimuth = np.deg2rad(pd.to_numeric(frame["azimuth_deg"], errors="coerce").fillna(0).to_numpy(float))
        distance = pd.to_numeric(frame["range_m"], errors="coerce").fillna(0).to_numpy(float)
        x = distance * np.sin(azimuth)
        y = distance * np.cos(azimuth)
        axis = axes[row_index]
        axis.plot(x, y, color="#2F6F8F", linewidth=1.5)
        axis.scatter([x[0]], [y[0]], color="#4C9F70", s=22, zorder=3, label="起点")
        axis.scatter([x[-1]], [y[-1]], color="#C65D3B", s=22, zorder=3, label="终点")
        axis.set_title(class_name, loc="left", fontsize=11, fontweight="bold")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        if row_index == 0:
            axis.legend(frameon=False, fontsize=9, loc="best")
        if row_index in {0, 3}:
            axis.set_ylabel("北向距离 / m", fontsize=10)
        if row_index >= 3:
            axis.set_xlabel("横向距离 / m", fontsize=10)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return chosen_manifest


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA attribution but CUDA is unavailable")

    split = load_grouped_split(args.split)
    records = load_trajectory_records(args.track_index, split, "val")
    dataset, cache = load_or_build(records, DEFAULT_TRAJECTORY_CACHE_ROOT, {"partition": "val", "method": "original"})
    sequence, physical, padding_mask, labels, ids = collect_tensors(dataset)
    model = load_b01_trajectory_branch(args.checkpoint).to(device).eval()
    baseline = macro_f1(model, sequence, physical, padding_mask, labels, ids, device, args.batch_size)

    rows = []
    for index, field in enumerate(TRACK_FEATURE_COLUMNS):
        if field not in EXCLUDED_SEQUENCE_FIELDS:
            rows.append(permutation_rows(baseline, index, field, "sequence", model, sequence, physical, padding_mask, labels, ids, args))
    for index, field in enumerate(PHYSICAL_FEATURE_COLUMNS):
        if field not in EXCLUDED_PHYSICAL_FIELDS:
            rows.append(permutation_rows(baseline, index, field, "physical", model, sequence, physical, padding_mask, labels, ids, args))

    rows.sort(key=lambda item: item["macro_f1_drop"], reverse=True)
    with (output / "permutation_importance.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["branch", "field", "name", "macro_f1", "accuracy", "macro_f1_drop", "accuracy_drop"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    make_figure(rows, output / "tr_feature_importance.png")
    selected_trajectory_examples = make_trajectory_figure(records, output / "tr_trajectory_feature_examples.png")
    summary = {
        "method": "single-seed permutation attribution",
        "interpretation": "A positive drop means the trained B01 branch depends on that input field on the fixed validation set; it is not a causal physical attribution.",
        "split": str(args.split.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "partition": "val", "trajectory_count": len(ids), "seed": args.seed, "device": str(device),
        "cache": cache, "baseline": baseline,
        "excluded_fields": {"sequence": sorted(EXCLUDED_SEQUENCE_FIELDS), "physical": sorted(EXCLUDED_PHYSICAL_FIELDS)},
        "trajectory_examples": selected_trajectory_examples,
        "ranking": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"output_dir": str(output), "baseline_macro_f1": baseline["macro_f1"], "top_fields": rows[:8]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

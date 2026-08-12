"""Generate structurally representative raw RD maps for all five classes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "radar_rd"))

from train import CLASS_NAMES, RDCache, SmallRDCNN, build_manifest, load_rd, split_frames  # noqa: E402
from analyze_bird_other_features import frame_features  # noqa: E402


CLASS_NAMES_ZH = ["无人机", "飞鸟", "空飘球", "杂波", "未知目标"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-features", type=Path, required=True,
                        help="JSON emitted by analyze_bird_other_features.py for the same checkpoint and split.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def input_image(frame, config: dict[str, object], cache: RDCache | None) -> np.ndarray:
    preprocessing = config.get("velocity_preprocessing", {})
    interval = preprocessing.get("common_interval_mps", [-90.0, 89.0])
    velocity_min = float(config.get("velocity_min", interval[0]))
    velocity_max = float(config.get("velocity_max", interval[1]))
    width = int(config.get("target_width", preprocessing.get("target_width", 900)))
    resampling = str(config.get("resampling", preprocessing.get("interpolation", "db_linear")))
    physical, observed = (cache.load(frame.path) if cache is not None else
                          load_rd(frame.path, velocity_min, velocity_max, width, resampling))
    physical = np.clip(physical, 0.0, 100.0)
    normalization = str(config.get("normalization", "global_z"))
    if normalization == "global_z":
        base = (physical - float(config["normalization_mean"])) / float(config["normalization_std"])
    elif normalization == "frame_z":
        base = (physical - physical.mean()) / max(float(physical.std()), 1e-6)
    elif normalization == "frame_robust":
        median = float(np.median(physical))
        spread = float(np.percentile(physical, 75) - np.percentile(physical, 25))
        base = (physical - median) / max(spread, 1e-6)
    elif normalization == "minmax":
        low, high = np.percentile(physical, [1, 99])
        base = np.clip((physical - low) / max(high - low, 1e-6), 0.0, 1.0)
    else:
        base = physical / 100.0
    channels = [base.astype(np.float32, copy=False)]
    input_mode = str(config.get("input_mode", "rd"))
    if input_mode == "rd_contrast":
        window = max(5, int(round(width * 0.04)))
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(physical, ((0, 0), (pad, pad)), mode="edge")
        prefix = np.cumsum(np.pad(padded, ((0, 0), (1, 0))), axis=1)
        local_mean = (prefix[:, window:] - prefix[:, :-window]) / float(window)
        contrast = (physical - local_mean) / max(float(config["normalization_std"]), 1e-6)
        channels.append(np.clip(contrast, -5.0, 5.0).astype(np.float32, copy=False))
    elif input_mode == "rd_mask":
        channels.append(np.broadcast_to(observed[None, :], base.shape).astype(np.float32))
    elif input_mode != "rd":
        raise ValueError(f"Unsupported input mode: {input_mode}")
    return np.stack(channels, axis=0)


@torch.inference_mode()
def predict_frames(model: SmallRDCNN, frames, config: dict[str, object], cache: RDCache | None,
                   batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    probabilities: dict[str, np.ndarray] = {}
    model.eval()
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        images = np.stack([input_image(frame, config, cache) for frame in chunk])
        batch = torch.from_numpy(images).to(device, non_blocking=True)
        values = torch.softmax(model(batch), dim=1).cpu().numpy()
        probabilities.update({frame.path: value for frame, value in zip(chunk, values)})
    return probabilities


STRUCTURE_KEYS = (
    "near_zero_energy_ratio", "response_width_mps", "local_contrast_peak",
    "range_peak_energy_ratio", "adjacent_profile_cosine",
)
FRAME_KEYS = (
    "near_zero_energy_ratio", "response_width_mps", "local_contrast_peak",
    "range_peak_energy_ratio",
)


def robust_scale(values: np.ndarray) -> float:
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return max(float(q75 - q25), float(values.std()), 1e-6)


def select_samples(frames, probabilities: dict[str, np.ndarray], feature_rows: list[dict[str, object]],
                   config: dict[str, object], cache: RDCache | None) -> list[dict[str, object]]:
    by_trajectory = defaultdict(list)
    for frame in frames:
        by_trajectory[frame.trajectory_id].append(frame)
    feature_by_trajectory = {str(row["trajectory_id"]): row for row in feature_rows}
    correct_by_class = defaultdict(list)
    trajectory_details: dict[str, tuple[np.ndarray, int, list]] = {}
    for trajectory_id, trajectory_frames in by_trajectory.items():
        trajectory_frames.sort(key=lambda frame: frame.path)
        mean_probability = np.mean([probabilities[frame.path] for frame in trajectory_frames], axis=0)
        label = trajectory_frames[0].label
        trajectory_details[trajectory_id] = (mean_probability, label, trajectory_frames)
        if int(mean_probability.argmax()) == label and trajectory_id in feature_by_trajectory:
            correct_by_class[label].append(trajectory_id)

    selected = []
    for label in range(len(CLASS_NAMES)):
        candidates = correct_by_class[label]
        if not candidates:
            raise RuntimeError(f"No correctly classified validation trajectory for {CLASS_NAMES[label]}")
        all_class_rows = [row for row in feature_rows if int(row["true"]) == label]
        medians = {key: float(np.median([float(row[key]) for row in all_class_rows])) for key in STRUCTURE_KEYS}
        scales = {key: robust_scale(np.asarray([float(row[key]) for row in all_class_rows], dtype=np.float64))
                  for key in STRUCTURE_KEYS}
        def structural_distance(trajectory_id: str) -> float:
            row = feature_by_trajectory[trajectory_id]
            return float(sum(abs(float(row[key]) - medians[key]) / scales[key] for key in STRUCTURE_KEYS))
        trajectory_id = min(candidates, key=lambda value: (structural_distance(value), str(value)))
        average, _, trajectory_frames = trajectory_details[trajectory_id]
        frame_rows = [(frame, frame_features(frame, config, cache)) for frame in trajectory_frames]
        frame_medians = {key: float(np.median([float(row[key]) for _, row in frame_rows])) for key in FRAME_KEYS}
        frame_scales = {key: robust_scale(np.asarray([float(row[key]) for _, row in frame_rows], dtype=np.float64))
                        for key in FRAME_KEYS}
        frame = min(frame_rows, key=lambda pair: (
            sum(abs(float(pair[1][key]) - frame_medians[key]) / frame_scales[key] for key in FRAME_KEYS),
            pair[0].path))[0]
        selected.append({
            "label_index": label,
            "class_name": CLASS_NAMES[label],
            "class_name_zh": CLASS_NAMES_ZH[label],
            "trajectory_id": trajectory_id,
            "frame_file": frame.path,
            "trajectory_probability": float(average[label]),
            "structural_distance_to_class_median": structural_distance(trajectory_id),
            "class_structure_medians": medians,
            "frame_probability": float(probabilities[frame.path][label]),
        })
    return selected


def render_figure(selected: list[dict[str, object]], config: dict[str, object], output: Path) -> None:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    width = int(config.get("target_width", 900))
    velocity_min = float(config.get("velocity_min", -90.0))
    velocity_max = float(config.get("velocity_max", 89.0))
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), constrained_layout=True)
    image_artist = None
    for index, record in enumerate(selected):
        axis = axes.flat[index]
        physical, _ = load_rd(record["frame_file"], velocity_min, velocity_max, width,
                              str(config.get("resampling", "db_linear")))
        physical = np.clip(physical, 0.0, 100.0)
        image_artist = axis.imshow(physical, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100,
                                   extent=(velocity_min, velocity_max, 0, physical.shape[0] - 1))
        axis.set_title(f"{record['class_name_zh']}  航迹 {record['trajectory_id']}\n"
                       f"航迹级{record['class_name_zh']}预测概率 {record['trajectory_probability']:.3f}")
        axis.set_xlabel("径向速度 / (m·s$^{-1}$)")
        axis.set_ylabel("相对距离单元")
        axis.set_xlim(velocity_min, velocity_max)
    axes.flat[-1].axis("off")
    colorbar = figure.colorbar(image_artist, ax=axes.ravel().tolist()[:-1], shrink=0.90, pad=0.02)
    colorbar.set_label("RD 幅度 / dB")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config = read_json(checkpoint.parent / "config.json")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    input_channels = 2 if str(config.get("input_mode")) in {"rd_contrast", "rd_mask"} else 1
    model = SmallRDCNN(input_channels=input_channels, head=str(config.get("model_head", "global")))
    model.load_state_dict(checkpoint_data["model_state"])
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    dataset_root = Path(str(config["dataset_root"]))
    frames = build_manifest(dataset_root)
    split = read_json(checkpoint.parent / "split.json")
    val_frames = split_frames(frames, split)["val"]
    cache_path = config.get("rd_cache")
    cache = None
    if cache_path:
        cache = RDCache(Path(str(cache_path)), val_frames,
                        velocity_min=float(config["velocity_min"]), velocity_max=float(config["velocity_max"]),
                        target_width=int(config["target_width"]), resampling=str(config["resampling"]))
    probabilities = predict_frames(model, val_frames, config, cache, args.batch_size, device)
    feature_rows = json.loads(args.trajectory_features.resolve().read_text(encoding="utf-8"))
    selected = select_samples(val_frames, probabilities, feature_rows, config, cache)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    render_figure(selected, config, output / "five_class_representative_rd.png")
    manifest = {
        "selection_rule": "For each true class, calculate the class-median RD structure from all validation trajectories. Among correctly classified trajectories, select the one with the minimum robust distance to that median across near-zero energy, velocity-response width, local contrast, range concentration, and adjacent-frame stability. Within it, select the frame closest to the trajectory's median frame structure.",
        "checkpoint": str(checkpoint),
        "split": "validation", "class_names": CLASS_NAMES, "samples": selected,
        "preprocessing": {key: config[key] for key in ("velocity_min", "velocity_max", "target_width", "resampling", "normalization", "input_mode")},
    }
    (output / "five_class_representative_rd_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Exploratory Grad-CAM attribution for the RD Bird/Other boundary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm, pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "radar_rd"))
from train import (  # noqa: E402
    CLASS_NAMES,
    RDCache,
    SmallRDCNN,
    build_manifest,
    load_rd,
    split_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames-per-trajectory", type=int, default=32)
    parser.add_argument("--max-trajectories-per-group", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_config(checkpoint: Path) -> dict[str, object]:
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    return config


def input_image(frame, config: dict[str, object], rd_cache: Optional[RDCache] = None) -> np.ndarray:
    preprocessing = config.get("velocity_preprocessing", {})
    interval = preprocessing.get("common_interval_mps", [-90.0, 89.0])
    velocity_min = float(config.get("velocity_min", interval[0]))
    velocity_max = float(config.get("velocity_max", interval[1]))
    target_width = int(config.get("target_width", preprocessing.get("target_width", 360)))
    resampling = str(config.get("resampling", preprocessing.get("interpolation", "db_linear")))
    if resampling == "linear_in_db":
        resampling = "db_linear"
    array, observed = (rd_cache.load(frame.path) if rd_cache is not None else
                       load_rd(frame.path, velocity_min, velocity_max, target_width, resampling))
    array = np.clip(array, 0.0, 100.0)
    physical = array
    normalization = str(config.get("normalization", "global_z"))
    if normalization == "global_z":
        array = (array - float(config.get("normalization_mean", config.get("mean", 0.0)))) / float(config.get("normalization_std", config.get("std", 1.0)))
    elif normalization == "frame_z":
        array = (array - float(array.mean())) / max(float(array.std()), 1e-6)
    elif normalization == "frame_robust":
        median = float(np.median(array)); spread = float(np.percentile(array, 75) - np.percentile(array, 25))
        array = (array - median) / max(spread, 1e-6)
    elif normalization == "minmax":
        low, high = np.percentile(array, [1, 99]); array = np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)
    else:
        array = array / 100.0
    channels = [array.astype(np.float32, copy=False)]
    input_mode = str(config.get("input_mode", "rd"))
    if input_mode == "rd_contrast":
        window = max(5, int(round(target_width * 0.04)))
        if window % 2 == 0:
            window += 1
        padding = window // 2
        padded = np.pad(physical, ((0, 0), (padding, padding)), mode="edge")
        prefix = np.cumsum(np.pad(padded, ((0, 0), (1, 0)), mode="constant"), axis=1)
        local_mean = (prefix[:, window:] - prefix[:, :-window]) / float(window)
        contrast = (physical - local_mean) / max(float(config.get("normalization_std", 1.0)), 1e-6)
        channels.append(np.clip(contrast, -5.0, 5.0).astype(np.float32, copy=False))
    elif input_mode == "rd_mask":
        channels.append(np.broadcast_to(observed[None, :], array.shape).astype(np.float32))
    else:
        if input_mode != "rd":
            raise ValueError(f"Attribution does not support input_mode={input_mode}")
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def frame_batches(frames, config, batch_size, rd_cache):
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        images = np.stack([input_image(frame, config, rd_cache) for frame in chunk], axis=0)
        yield chunk, torch.from_numpy(images)


def frame_predictions(model, frames, config, batch_size, rd_cache, device):
    probabilities = {}
    model.eval()
    with torch.inference_mode():
        for chunk, images in frame_batches(frames, config, batch_size, rd_cache):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.update({frame.path: prob.tolist() for frame, prob in zip(chunk, probs)})
    return probabilities


def trajectory_predictions(frames, probabilities):
    grouped = defaultdict(list)
    labels = {}
    paths = defaultdict(list)
    for frame in frames:
        grouped[frame.trajectory_id].append(np.asarray(probabilities[frame.path], dtype=np.float64))
        labels[frame.trajectory_id] = frame.label
        paths[frame.trajectory_id].append(frame)
    result = {}
    for trajectory_id, probs in grouped.items():
        mean_probability = np.mean(probs, axis=0)
        result[trajectory_id] = {
            "true": labels[trajectory_id], "pred": int(np.argmax(mean_probability)),
            "probabilities": mean_probability.tolist(), "frames": paths[trajectory_id],
        }
    return result


def trajectories_from_saved_metrics(frames, metrics: dict[str, object]):
    grouped_frames = defaultdict(list)
    labels = {}
    for frame in frames:
        grouped_frames[frame.trajectory_id].append(frame)
        labels[frame.trajectory_id] = frame.label
    predictions = dict(labels)
    for pair, trajectory_ids in metrics.get("confusion_cases", {}).items():
        _actual, predicted = (int(value) for value in pair.split(":"))
        for trajectory_id in trajectory_ids:
            predictions[str(trajectory_id)] = predicted
    return {trajectory_id: {"true": labels[trajectory_id], "pred": predictions[trajectory_id],
                            "frames": trajectory_frames}
            for trajectory_id, trajectory_frames in grouped_frames.items()}


def gradcam(model, image: torch.Tensor, target: int, layer, activations, gradients, device) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    logits = model(image[None].to(device))
    score = logits[0, target]
    score.backward()
    activation = activations["value"]
    gradient = gradients["value"]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    # Some checkpoints produce exclusively negative class evidence at this
    # layer, making ReLU Grad-CAM identically zero. Magnitude attribution keeps
    # both excitatory and inhibitory sensitivity; occlusion below supplies the
    # direction of the probability effect.
    cam = torch.abs((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    scale = float(cam.max())
    return cam / scale if scale > 1e-8 else np.zeros_like(cam)


def save_heatmap(path: Path, heatmap: np.ndarray, title: str, width: int) -> None:
    figure, axis = plt.subplots(figsize=(10, 3.8), dpi=150)
    axis.imshow(heatmap, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0,
                extent=(-90, 89, 0, heatmap.shape[0]))
    axis.set_xlabel("Vr (m/s)")
    axis.set_ylabel("RD row")
    axis.set_title(f"{title} · 31×{width}")
    figure.colorbar(axis.images[0], ax=axis, label="Gradient-activation magnitude")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def summarize_heatmap(heatmap: np.ndarray) -> dict[str, object]:
    mass = np.maximum(heatmap, 0.0)
    column = mass.mean(axis=0)
    row = mass.mean(axis=1)
    total = float(column.sum())
    if total <= 1e-8:
        return {"velocity_center_mps": None, "velocity_abs_center_mps": None,
                "velocity_deciles": [0.0] * 10, "row_quartiles": [0.0] * 4}
    vr = np.linspace(-90.0, 89.0, heatmap.shape[1])
    deciles = [float(chunk.sum() / total) for chunk in np.array_split(column, 10)]
    rows = [float(chunk.sum() / max(float(row.sum()), 1e-8)) for chunk in np.array_split(row, 4)]
    return {"velocity_center_mps": float((column * vr).sum() / total),
            "velocity_abs_center_mps": float((column * np.abs(vr)).sum() / total),
            "velocity_deciles": deciles, "row_quartiles": rows,
            "peak_velocity_mps": float(vr[int(np.argmax(column))])}


@torch.inference_mode()
def mean_target_probability(model, images: list[torch.Tensor], target: int, device,
                            batch_size: int, mask: Optional[tuple[str, object]] = None) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start:start + batch_size]).to(device)
        if mask is not None:
            kind, value = mask
            batch = batch.clone()
            if kind == "channel":
                batch[:, int(value)] = 0.0
            elif kind == "velocity":
                batch[:, :, :, list(value)] = 0.0
            elif kind == "row":
                batch[:, :, list(value), :] = 0.0
            else:
                raise ValueError(f"Unknown occlusion mask: {kind}")
        probabilities = torch.softmax(model(batch), dim=1)[:, target]
        total += float(probabilities.sum().cpu())
        count += int(probabilities.numel())
    return total / max(count, 1)


def occlusion_sensitivity(model, images: list[torch.Tensor], target: int, config: dict[str, object],
                          device, batch_size: int) -> dict[str, object]:
    if not images:
        return {}
    base = mean_target_probability(model, images, target, device, batch_size)
    width = images[0].shape[-1]
    rows = images[0].shape[-2]
    velocity_min = float(config.get("velocity_min", -90.0))
    velocity_max = float(config.get("velocity_max", 89.0))
    velocity = np.linspace(velocity_min, velocity_max, width)
    velocity_masks = {
        "central_abs_le_10_mps": np.flatnonzero(np.abs(velocity) <= 10.0).tolist(),
        "mid_abs_10_to_30_mps": np.flatnonzero((np.abs(velocity) > 10.0) & (np.abs(velocity) <= 30.0)).tolist(),
        "outer_abs_gt_30_mps": np.flatnonzero(np.abs(velocity) > 30.0).tolist(),
    }
    channel_results = {}
    for channel in range(images[0].shape[0]):
        probability = mean_target_probability(model, images, target, device, batch_size,
                                              ("channel", channel))
        channel_results[str(channel)] = {"occluded_probability": probability,
                                         "probability_drop": base - probability}
    velocity_results = {}
    for name, columns in velocity_masks.items():
        probability = mean_target_probability(model, images, target, device, batch_size,
                                              ("velocity", columns))
        velocity_results[name] = {"occluded_probability": probability,
                                  "probability_drop": base - probability}
    row_results = {}
    for index, indices in enumerate(np.array_split(np.arange(rows), 4)):
        probability = mean_target_probability(model, images, target, device, batch_size,
                                              ("row", indices.tolist()))
        row_results[f"quartile_{index + 1}"] = {"occluded_probability": probability,
                                                "probability_drop": base - probability}
    return {"target_class": CLASS_NAMES[target], "base_probability": base,
            "channel_occlusion": channel_results, "velocity_occlusion": velocity_results,
            "row_occlusion": row_results}


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    input_mode = str(config.get("input_mode", "rd"))
    input_channels = 1 if input_mode == "rd" else 2
    model = SmallRDCNN(input_channels=input_channels, head=str(config.get("model_head", "global")))
    model.load_state_dict(checkpoint_data["model_state"])
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root is not None else Path(str(config["dataset_root"]))
    frames = build_manifest(dataset_root)
    split = json.loads((checkpoint.parent / "split.json").read_text(encoding="utf-8"))
    val_frames = split_frames(frames, split)["val"]
    rd_cache = None
    cache_value = config.get("rd_cache")
    if cache_value:
        rd_cache = RDCache(Path(str(cache_value)), val_frames,
                           velocity_min=float(config.get("velocity_min", -90.0)),
                           velocity_max=float(config.get("velocity_max", 89.0)),
                           target_width=int(config.get("target_width", 360)),
                           resampling=str(config.get("resampling", "db_linear")))
    validation_path = checkpoint.parent / "validation_best.json"
    validation_metrics = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    if validation_metrics.get("confusion_cases") is not None:
        trajectories = trajectories_from_saved_metrics(val_frames, validation_metrics)
        grouping_source = "validation_best.confusion_cases"
    else:
        probabilities = frame_predictions(model, val_frames, config, args.batch_size, rd_cache, device)
        trajectories = trajectory_predictions(val_frames, probabilities)
        grouping_source = "model_recomputed"
    groups = {"bird_correct": (1, 1), "bird_to_other": (1, 4),
              "other_correct": (4, 4), "other_to_bird": (4, 1),
              "other_to_clutter": (4, 3), "bird_to_balloon": (1, 2),
              "other_to_balloon": (4, 2)}
    grouped_ids = {name: [tid for tid, item in trajectories.items() if (item["true"], item["pred"]) == pair]
                   for name, pair in groups.items()}

    activations, gradients = {}, {}
    # Attribute the feature map immediately before adaptive pooling. Hooking
    # the convolution before BatchNorm/GELU can make spatially averaged
    # gradients cancel to zero even when the model is highly sensitive.
    layer = model.features[-1]
    handles = [layer.register_forward_hook(lambda _m, _i, out: activations.update(value=out)),
               layer.register_full_backward_hook(lambda _m, _gi, go: gradients.update(value=go[0]))]
    summary = {"checkpoint": str(checkpoint), "class_names": CLASS_NAMES,
               "device": str(device), "input_mode": input_mode, "grouping_source": grouping_source,
               "validation_trajectory_count": len(trajectories), "groups": {}}
    rng = np.random.default_rng(42)
    try:
        for name, ids in grouped_ids.items():
            selected_ids = ids if len(ids) <= args.max_trajectories_per_group else list(
                rng.choice(ids, args.max_trajectories_per_group, replace=False))
            heatmaps = []
            sampled_images = []
            for trajectory_id in selected_ids:
                frames_for_trajectory = trajectories[trajectory_id]["frames"]
                if len(frames_for_trajectory) > args.max_frames_per_trajectory:
                    indices = np.linspace(0, len(frames_for_trajectory) - 1, args.max_frames_per_trajectory).astype(int)
                    frames_for_trajectory = [frames_for_trajectory[index] for index in indices]
                target = trajectories[trajectory_id]["pred"]
                for frame in frames_for_trajectory:
                    image = torch.from_numpy(input_image(frame, config, rd_cache))
                    sampled_images.append(image)
                    heatmaps.append(gradcam(model, image, target, layer, activations, gradients, device))
            if heatmaps:
                aggregate = np.mean(heatmaps, axis=0)
                np.save(output / f"{name}.npy", aggregate)
                save_heatmap(output / f"{name}.png", aggregate, name,
                             int(config.get("target_width", config.get("velocity_preprocessing", {}).get("target_width", 360))))
                attribution = summarize_heatmap(aggregate)
            else:
                attribution = {"velocity_center_mps": None, "velocity_abs_center_mps": None}
            target = groups[name][1]
            summary["groups"][name] = {"trajectory_count": len(ids), "sampled_trajectory_count": len(selected_ids),
                                        "sampled_frame_count": len(heatmaps), "trajectory_ids": selected_ids,
                                        "attribution": attribution,
                                        "occlusion": occlusion_sensitivity(model, sampled_images, target, config,
                                                                            device, args.batch_size)}
    finally:
        for handle in handles:
            handle.remove()
    summary["trajectory_confusion_pairs"] = {name: len(ids) for name, ids in grouped_ids.items()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

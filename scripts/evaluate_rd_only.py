"""Evaluate an existing RD checkpoint with its original split and preprocessing.

This is deliberately separate from radar_rd.train: checkpoint evaluation must
not rebuild a split, recompute normalization statistics, or update a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_rd.train import (  # noqa: E402
    CLASS_NAMES,
    RDCache,
    RDDataset,
    SmallRDCNN,
    build_manifest,
    metrics_from_trajectory_probabilities,
    trajectory_decision_records,
    read_json_file,
    split_frames,
    trajectory_table,
    write_json,
    write_progress,
)
from radar_fusion.partition_augmentation import expand_rd_frames, validate_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--rd-cache", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--partition-augmentation-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition-augmentation-targets-train", type=int, nargs=5, default=None)
    parser.add_argument("--partition-augmentation-targets-val", type=int, nargs=5, default=None)
    parser.add_argument("--partition-augmentation-targets-test", type=int, nargs=5, default=None)
    parser.add_argument("--augment-existing", action="store_true",
                        help="Keep the existing original evaluation and only append the virtual-trajectory diagnostic")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_settings(source_config: dict[str, object], checkpoint: dict[str, object]) -> dict[str, object]:
    velocity = source_config.get("velocity_preprocessing")
    velocity = velocity if isinstance(velocity, dict) else {}
    settings = {
        "velocity_min": source_config.get("velocity_min", velocity.get("common_interval_mps", [-90.0, 89.0])[0]),
        "velocity_max": source_config.get("velocity_max", velocity.get("common_interval_mps", [-90.0, 89.0])[1]),
        "target_width": source_config.get("target_width", velocity.get("target_width", 360)),
        "resampling": source_config.get("resampling", velocity.get("interpolation", "db_linear")),
        "normalization": source_config.get("normalization", "global_z"),
        "input_mode": source_config.get("input_mode", "rd"),
        "model_head": source_config.get("model_head", "global"),
        "mean": checkpoint.get("mean", source_config.get("normalization_mean")),
        "std": checkpoint.get("std", source_config.get("normalization_std")),
    }
    settings["resampling"] = {"linear_in_db": "db_linear", "linear_in_power": "power_linear"}.get(
        str(settings["resampling"]), settings["resampling"]
    )
    required = ("velocity_min", "velocity_max", "target_width", "resampling", "normalization", "input_mode", "model_head", "mean", "std")
    missing = [key for key in required if settings.get(key) is None]
    if missing:
        raise ValueError(f"checkpoint source configuration lacks: {', '.join(missing)}")
    settings["velocity_min"] = float(settings["velocity_min"])
    settings["velocity_max"] = float(settings["velocity_max"])
    settings["target_width"] = int(settings["target_width"])
    settings["mean"] = float(settings["mean"])
    settings["std"] = max(float(settings["std"]), 1e-6)
    return settings


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    source_dir = checkpoint_path.parent
    source_config = read_json_file(source_dir / "config.json", {})
    split = read_json_file(source_dir / "split.json", {})
    if not isinstance(source_config, dict) or not isinstance(split, dict) or not split:
        raise ValueError("RD checkpoint must be inside a training artifact containing config.json and split.json")
    if set(split.values()) - {"train", "val", "test"}:
        raise ValueError("source split.json contains an unsupported partition")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("RD checkpoint lacks model_state")
    settings = source_settings(source_config, checkpoint)

    frames = build_manifest(args.dataset_root.expanduser().resolve())
    if set(split) != set(trajectory_table(frames)):
        raise ValueError("dataset trajectories do not match the checkpoint source split")
    partitions = split_frames(frames, {str(key): str(value) for key, value in split.items()})
    eval_frames = partitions[args.partition]
    if not eval_frames:
        raise ValueError(f"source split has no {args.partition} RD frames")
    requested = {"train": args.partition_augmentation_targets_train, "val": args.partition_augmentation_targets_val, "test": args.partition_augmentation_targets_test}[args.partition]
    augmented_frames, augmented_manifest = expand_rd_frames(eval_frames, partition=args.partition, targets=validate_targets(requested, eval_frames), seed=42) if args.partition_augmentation_diagnostics else (eval_frames, None)
    cache = None
    if args.rd_cache is not None:
        cache = RDCache(
            # Diagnostic virtual tracks reuse source paths.  Cache entries are
            # physical frames only, and any held-out/cache-miss frame follows
            # the identical deterministic loader path.
            args.rd_cache, eval_frames,
            velocity_min=settings["velocity_min"], velocity_max=settings["velocity_max"],
            target_width=settings["target_width"], resampling=settings["resampling"],
            allow_missing=True,
        )

    device = torch.device(args.device)
    dataset = RDDataset(
        eval_frames, settings["mean"], settings["std"],
        velocity_min=settings["velocity_min"], velocity_max=settings["velocity_max"],
        target_width=settings["target_width"], resampling=settings["resampling"],
        normalization=settings["normalization"], input_mode=settings["input_mode"],
        augmentation="off", rd_cache=cache,
        derived_cache_dir=PROJECT_ROOT / "cache" / "rd_partition_augmented",
    )
    augmented_dataset = RDDataset(
        augmented_frames, settings["mean"], settings["std"],
        velocity_min=settings["velocity_min"], velocity_max=settings["velocity_max"],
        target_width=settings["target_width"], resampling=settings["resampling"],
        normalization=settings["normalization"], input_mode=settings["input_mode"],
        augmentation="off", rd_cache=cache,
        derived_cache_dir=PROJECT_ROOT / "cache" / "rd_partition_augmented",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    augmented_loader = DataLoader(augmented_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    channels = 1 if settings["input_mode"] == "rd" else 2
    model = SmallRDCNN(input_channels=channels, head=settings["model_head"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not args.augment_existing:
        raise FileExistsError(f"Output directory must be empty: {output}")
    if args.augment_existing:
        if not args.partition_augmentation_diagnostics:
            raise ValueError("--augment-existing requires partition augmentation diagnostics")
        existing_config = read_json_file(output / "config.json", {})
        if not isinstance(existing_config, dict) or not existing_config:
            raise ValueError("Existing evaluation is missing config.json")
        augmented_metrics_path = output / ("test_augmented_trajectory_metrics.json" if args.partition == "test" else "validation_augmented_best.json")
        augmented_decisions_path = output / ("trajectory_decisions_test_augmented.jsonl" if args.partition == "test" else "trajectory_decisions_augmented.jsonl")
        if augmented_metrics_path.exists() or augmented_decisions_path.exists():
            raise ValueError("This evaluation already contains the augmentation diagnostic")
        write_progress(output, {"phase": "augmented_diagnostic", "batch": 0,
                                "total_batches": len(augmented_loader), "percent": 0.0,
                                "stage": "virtual_trajectory_inference"})
        augmented_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
        augmented_labels: dict[str, int] = {}
        for batch_index, (images, targets, trajectory_ids) in enumerate(augmented_loader, start=1):
            batch_probs = torch.softmax(model(images.to(device, non_blocking=True)), dim=1).cpu().numpy()
            for trajectory_id, target, values in zip(trajectory_ids, targets.tolist(), batch_probs):
                augmented_probabilities[str(trajectory_id)].append(values)
                augmented_labels[str(trajectory_id)] = int(target)
            write_progress(output, {"phase": "augmented_diagnostic", "batch": batch_index,
                                    "total_batches": len(augmented_loader),
                                    "percent": 100.0 * batch_index / max(len(augmented_loader), 1),
                                    "stage": "virtual_trajectory_inference"})
        augmented_metrics = metrics_from_trajectory_probabilities(augmented_probabilities, augmented_labels)
        augmented_decisions = trajectory_decision_records(augmented_probabilities, augmented_labels)
        augmented_decisions_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in augmented_decisions), encoding="utf-8"
        )
        write_json(augmented_metrics_path, augmented_metrics)
        write_json(output / "partition_augmentation_manifest.json", {"enabled": True, "partition": augmented_manifest})
        supplement = {
            "kind": "partition_augmentation_diagnostic",
            "partition": args.partition,
            "completed_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "trajectory_count": augmented_metrics.get("trajectory_count", 0),
        }
        write_json(output / "augmentation_supplement.json", supplement)
        merged_config = dict(existing_config)
        merged_config["partition_augmentation_diagnostics"] = {"enabled": True, "partition": augmented_manifest}
        merged_config["augmentation_supplement"] = supplement
        write_json(output / "config.json", merged_config)
        write_progress(output, {"phase": "complete", "percent": 100.0, "augmentation_supplement": True})
        print(json.dumps({"augmentation": True, "accuracy": augmented_metrics["accuracy"],
                          "macro_f1": augmented_metrics["macro_f1"], "output_dir": str(output)}, ensure_ascii=False, indent=2))
        return
    config = {
        "experiment_type": "rd_checkpoint_eval",
        "experiment_label": "RD checkpoint original-split evaluation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "source_output_dir": str(source_dir),
        "source_split": str(source_dir / "split.json"),
        "source_split_sha256": file_sha256(source_dir / "split.json"),
        "partition": args.partition,
        "batch_size_rd": args.batch_size,
        "workers": args.workers,
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "rd_cache": str(cache.cache_dir) if cache is not None else "",
        "device": str(device),
        "test_deferred": args.partition != "test",
        "evaluation_preprocessing": settings,
        "partition_augmentation_diagnostics": {"enabled": bool(args.partition_augmentation_diagnostics), "partition": augmented_manifest},
    }
    write_json(output / "config.json", config)
    write_progress(output, {"phase": "rd_inference", "batch": 0, "total_batches": len(loader), "percent": 0.0})

    probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, int] = {}
    loss_sum, count = 0.0, 0
    for batch_index, (images, targets, trajectory_ids) in enumerate(loader, start=1):
        logits = model(images.to(device, non_blocking=True))
        loss_sum += float(nn.functional.cross_entropy(logits, targets.to(device, non_blocking=True), reduction="sum").cpu())
        count += int(targets.numel())
        batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
        for trajectory_id, target, values in zip(trajectory_ids, targets.tolist(), batch_probs):
            probabilities[str(trajectory_id)].append(values)
            labels[str(trajectory_id)] = int(target)
        write_progress(output, {
            "phase": "rd_inference", "batch": batch_index, "total_batches": len(loader),
            "percent": 100.0 * batch_index / max(len(loader), 1), "loss": loss_sum / max(count, 1),
        })

    metrics = metrics_from_trajectory_probabilities(probabilities, labels)
    metrics["loss"] = loss_sum / max(count, 1)
    augmented_metrics = None
    augmented_decisions = []
    if args.partition_augmentation_diagnostics:
        write_progress(output, {
            "phase": "augmented_diagnostic", "batch": 0,
            "total_batches": len(augmented_loader), "percent": 0.0,
            "stage": "virtual_trajectory_inference",
        })
        augmented_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
        augmented_labels: dict[str, int] = {}
        for batch_index, (images, targets, trajectory_ids) in enumerate(augmented_loader, start=1):
            batch_probs = torch.softmax(model(images.to(device, non_blocking=True)), dim=1).cpu().numpy()
            for trajectory_id, target, values in zip(trajectory_ids, targets.tolist(), batch_probs):
                augmented_probabilities[str(trajectory_id)].append(values)
                augmented_labels[str(trajectory_id)] = int(target)
            write_progress(output, {
                "phase": "augmented_diagnostic", "batch": batch_index,
                "total_batches": len(augmented_loader),
                "percent": 100.0 * batch_index / max(len(augmented_loader), 1),
                "stage": "virtual_trajectory_inference",
            })
        augmented_metrics = metrics_from_trajectory_probabilities(augmented_probabilities, augmented_labels)
        augmented_decisions = trajectory_decision_records(augmented_probabilities, augmented_labels)
    write_progress(output, {
        "phase": "saving_results", "percent": 0.0,
        "stage": "writing_metrics_decisions",
    })
    decisions = []
    for trajectory_id in sorted(probabilities, key=str):
        average = np.mean(probabilities[trajectory_id], axis=0)
        prediction = int(average.argmax())
        truth = labels[trajectory_id]
        decisions.append({
            "trajectory_id": trajectory_id,
            "true_class": truth,
            "true_label": CLASS_NAMES[truth],
            "rd_prediction": prediction,
            "rd_prediction_label": CLASS_NAMES[prediction],
            "rd_probabilities": average.tolist(),
            "rd_frame_count": len(probabilities[trajectory_id]),
        })
    metrics_path = output / ("test_trajectory_metrics.json" if args.partition == "test" else "validation_best.json")
    write_json(metrics_path, metrics)
    if augmented_metrics is not None:
        write_json(output / ("test_augmented_trajectory_metrics.json" if args.partition == "test" else "validation_augmented_best.json"), augmented_metrics)
        (output / ("trajectory_decisions_test_augmented.jsonl" if args.partition == "test" else "trajectory_decisions_augmented.jsonl")).write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in augmented_decisions), encoding="utf-8"
        )
        write_json(output / "partition_augmentation_manifest.json", {"enabled": True, "partition": augmented_manifest})
    with (output / "trajectory_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    write_json(output / "ablation_complete.json", {
        "best_epoch": config["checkpoint_epoch"], "best_val_macro_f1": metrics["macro_f1"],
        "test_deferred": args.partition != "test",
    })
    write_progress(output, {"phase": "complete", "percent": 100.0, "checkpoint_epoch": config["checkpoint_epoch"]})
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "trajectory_count": metrics["trajectory_count"], "output_dir": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

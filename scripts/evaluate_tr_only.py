"""Evaluate an existing TR checkpoint on the fixed F split.

This is intentionally not a training script.  New training jobs use
train_tr_only.py and are exposed as TR-only 训练 in the monitor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
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
from radar_fusion.model import CLASS_NAMES, load_b01_trajectory_branch, load_checkpoint_metadata  # noqa: E402
from radar_fusion.reporting import classification_metrics  # noqa: E402
from radar_fusion.ml_track import clean_train_features, feature_matrix, load_bundle  # noqa: E402
from radar_fusion.partition_augmentation import expand_trajectory_records, validate_targets  # noqa: E402
from radar_fusion.trajectory_cache import DEFAULT_TRAJECTORY_CACHE_ROOT, load_or_build  # noqa: E402


DEFAULT_SPLIT = Path(r"K:\radar\main\data\manifests\cq08_grouped_split_f.json")
DEFAULT_TRACK_INDEX = Path(r"K:\radar\main\data\processed\expert1_track_index.csv")
DEFAULT_CHECKPOINT = Path(
    r"K:\radar\main\artifacts\f_protocol\20260728-183147\b01_transformer_seed42\best_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--track-index", type=Path, default=DEFAULT_TRACK_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--partition", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--partition-augmentation-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition-augmentation-method", choices=("perturbation", "smote"), default="perturbation")
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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_ml_checkpoint(args, split, records, output):
    bundle = load_bundle(args.checkpoint)
    x, y, names, cache = feature_matrix(records, partition=args.partition)
    median = np.asarray(bundle.get("median", np.zeros(x.shape[1])), dtype=np.float64)
    x = np.where(np.isfinite(x), x, median)
    model = bundle["model"]
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    rows = [{"trajectory_id": record.trajectory_id, "true_class": int(record.label),
             "true_label": CLASS_NAMES[int(record.label)], "tr_probabilities": prob.tolist(),
             "prediction": int(prediction), "prediction_label": CLASS_NAMES[int(prediction)],
             "implementation": "ml", "checkpoint": str(args.checkpoint.resolve())}
            for record, prob, prediction in zip(records, probabilities, predictions)]
    metrics = classification_metrics(rows, "prediction")
    metrics_path = output / ("test_trajectory_metrics.json" if args.partition == "test" else "validation_best.json")
    write_json(metrics_path, metrics)
    (output / "trajectory_decisions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(output / "config.json", {"experiment_type": "tr_checkpoint_eval", "tr_implementation": "ml",
        "checkpoint": str(args.checkpoint.resolve()), "partition": args.partition,
        "feature_names": names, "feature_cache": cache, "test_deferred": args.partition != "test"})
    write_json(output / "progress.json", {"phase": "complete", "percent": 100.0,
        "implementation": "ml", "partition": args.partition})
    return metrics


@torch.inference_mode()
def infer_records(model, loader, device, progress_output=None):
    result = []
    total_batches = len(loader)
    if progress_output is not None:
        write_json(progress_output / "progress.json", {"phase": "augmented_diagnostic", "batch": 0,
                                                         "total_batches": total_batches, "percent": 0.0,
                                                         "stage": "virtual_trajectory_inference"})
    for batch_index, batch in enumerate(loader, start=1):
        logits = model(batch.sequence.to(device, non_blocking=True), batch.physical.to(device, non_blocking=True), batch.padding_mask.to(device, non_blocking=True)).cpu()
        probabilities = torch.softmax(logits, dim=-1).numpy(); predictions = logits.argmax(dim=-1).tolist()
        for trajectory_id, label, values, probs, prediction in zip(batch.trajectory_ids, batch.labels.tolist(), logits.numpy(), probabilities, predictions):
            result.append({"trajectory_id": trajectory_id, "true_class": int(label), "true_label": CLASS_NAMES[int(label)], "tr_logits": values.tolist(), "tr_probabilities": probs.tolist(), "prediction": int(prediction), "prediction_label": CLASS_NAMES[int(prediction)]})
        if progress_output is not None:
            write_json(progress_output / "progress.json", {"phase": "augmented_diagnostic", "batch": batch_index,
                                                             "total_batches": total_batches,
                                                             "percent": 100.0 * batch_index / max(total_batches, 1),
                                                             "stage": "virtual_trajectory_inference"})
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split = load_grouped_split(args.split)
    records = load_trajectory_records(args.track_index, split, args.partition)
    if args.checkpoint.suffix.lower() in {".joblib", ".pkl"}:
        evaluate_ml_checkpoint(args, split, records, output)
        return
    requested = {"train": args.partition_augmentation_targets_train, "val": args.partition_augmentation_targets_val, "test": args.partition_augmentation_targets_test}[args.partition]
    augmented_records, augmented_manifest = expand_trajectory_records(
        records, partition=args.partition, targets=validate_targets(requested, records), seed=42,
        method=args.partition_augmentation_method,
    ) if args.partition_augmentation_diagnostics else (records, None)
    write_json(output / "progress.json", {
        "phase": "preparing", "stage": "loading_tr_feature_cache", "percent": 0.0,
    })
    dataset, original_cache = load_or_build(
        records, DEFAULT_TRAJECTORY_CACHE_ROOT,
        {"partition": args.partition, "method": "original"},
    )
    augmented_context = (
        {"partition_augmentation": augmented_manifest["cache_parameters"]}
        if augmented_manifest else {"partition": args.partition, "method": "original"}
    )
    augmented_dataset, augmented_cache = load_or_build(
        augmented_records, DEFAULT_TRAJECTORY_CACHE_ROOT, augmented_context
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_trajectories,
        pin_memory=args.device.startswith("cuda"),
    )
    augmented_loader = DataLoader(augmented_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_trajectories, pin_memory=args.device.startswith("cuda"))
    device = torch.device(args.device)
    model = load_b01_trajectory_branch(args.checkpoint).to(device).eval()
    checkpoint = load_checkpoint_metadata(args.checkpoint)
    if args.augment_existing:
        if not args.partition_augmentation_diagnostics:
            raise ValueError("--augment-existing requires partition augmentation diagnostics")
        existing_config = read_json(output / "config.json")
        if not isinstance(existing_config, dict) or not existing_config:
            raise ValueError("Existing evaluation is missing config.json")
        augmented_path = output / ("test_augmented_trajectory_metrics.json" if args.partition == "test" else "validation_augmented_best.json")
        augmented_decisions_path = output / "trajectory_decisions_augmented.jsonl"
        if augmented_path.exists() or augmented_decisions_path.exists():
            raise ValueError("This evaluation already contains the augmentation diagnostic")
        augmented_out = infer_records(model, augmented_loader, device, output)
        augmented_metrics = classification_metrics(augmented_out, "prediction")
        augmented_path.write_text(json.dumps(augmented_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        augmented_decisions_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in augmented_out), encoding="utf-8")
        write_json(output / "partition_augmentation_manifest.json", {"enabled": True, "partition": augmented_manifest})
        supplement = {"kind": "partition_augmentation_diagnostic", "partition": args.partition,
                      "completed_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                      "trajectory_count": augmented_metrics.get("trajectory_count", 0)}
        write_json(output / "augmentation_supplement.json", supplement)
        merged_config = dict(existing_config)
        merged_config["partition_augmentation_diagnostics"] = {"enabled": True, "method": args.partition_augmentation_method, "partition": augmented_manifest}
        merged_config["tr_feature_cache"] = {"original": original_cache, "augmented": augmented_cache}
        merged_config["augmentation_supplement"] = supplement
        write_json(output / "config.json", merged_config)
        write_json(output / "progress.json", {"phase": "complete", "percent": 100.0, "augmentation_supplement": True})
        print(json.dumps({"augmentation": True, "accuracy": augmented_metrics["accuracy"],
                          "macro_f1": augmented_metrics["macro_f1"], "output_dir": str(output)}, ensure_ascii=False, indent=2))
        return
    config = {
        "experiment_type": "tr_checkpoint_eval",
        "experiment_label": "B01 TR checkpoint fixed F split evaluation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "split": str(args.split.resolve()),
        "split_sha256": file_sha256(args.split),
        "partition": args.partition,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": str(device),
        "test_deferred": args.partition != "test",
        "partition_augmentation_method": args.partition_augmentation_method,
        "partition_augmentation_diagnostics": {"enabled": bool(args.partition_augmentation_diagnostics), "method": args.partition_augmentation_method, "partition": augmented_manifest},
        "tr_feature_cache": {"original": original_cache, "augmented": augmented_cache},
    }
    write_json(output / "config.json", config)
    write_json(output / "progress.json", {
        "phase": "trajectory_inference", "batch": 0,
        "total_batches": len(loader), "percent": 0.0,
    })
    records_out: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader, start=1):
        logits = model(
            batch.sequence.to(device, non_blocking=True),
            batch.physical.to(device, non_blocking=True),
            batch.padding_mask.to(device, non_blocking=True),
        ).cpu()
        probabilities = torch.softmax(logits, dim=-1).numpy()
        predictions = logits.argmax(dim=-1).tolist()
        for trajectory_id, label, values, probs, prediction in zip(
            batch.trajectory_ids, batch.labels.tolist(), logits.numpy(), probabilities, predictions
        ):
            records_out.append({
                "trajectory_id": trajectory_id,
                "true_class": int(label),
                "true_label": CLASS_NAMES[int(label)],
                "tr_logits": values.tolist(),
                "tr_probabilities": probs.tolist(),
                "prediction": int(prediction),
                "prediction_label": CLASS_NAMES[int(prediction)],
            })
        write_json(output / "progress.json", {
            "phase": "trajectory_inference", "batch": batch_index,
            "total_batches": len(loader),
            "percent": 100.0 * batch_index / max(len(loader), 1),
        })
    # Virtual diagnostic IDs contain a provenance suffix and are not numeric.
    records_out.sort(key=lambda item: str(item["trajectory_id"]))
    metrics = classification_metrics(records_out, "prediction")
    augmented_out = infer_records(model, augmented_loader, device, output) if args.partition_augmentation_diagnostics else []
    augmented_metrics = classification_metrics(augmented_out, "prediction") if augmented_out else None
    write_json(output / "config.json", config)
    metrics_path = output / ("test_trajectory_metrics.json" if args.partition == "test" else "validation_best.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if augmented_metrics is not None:
        augmented_path = output / ("test_augmented_trajectory_metrics.json" if args.partition == "test" else "validation_augmented_best.json")
        augmented_path.write_text(json.dumps(augmented_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "trajectory_decisions_augmented.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in augmented_out), encoding="utf-8")
        write_json(output / "partition_augmentation_manifest.json", {"enabled": True, "partition": augmented_manifest})
    (output / "trajectory_decisions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records_out), encoding="utf-8"
    )
    (output / "ablation_complete.json").write_text(
        json.dumps({"best_epoch": config["checkpoint_epoch"], "best_val_macro_f1": metrics["macro_f1"],
                    "test_deferred": args.partition != "test"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_json(output / "progress.json", {"phase": "complete", "percent": 100.0})
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "trajectory_count": metrics["trajectory_count"],
        "confusion_matrix": metrics["confusion_matrix"],
        "output_dir": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

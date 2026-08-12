"""Fit a small TR-RD quality gate from frozen checkpoint predictions.

This is the lightweight, modular gate path: TR and RD are never retrained.
The selected calibration partition is used once to write reusable branch-score
records, then only the ~700-parameter gate is optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_soft_cascade import infer_rd, infer_tr, write_progress  # noqa: E402
from generate_fusion_oof import train_gate_inline  # noqa: E402
from radar_fusion.data import load_grouped_split  # noqa: E402
from radar_fusion.model import CLASS_NAMES, SoftCascadeFusion, load_checkpoint_metadata  # noqa: E402
from radar_fusion.reporting import classification_metrics, save_fusion_report  # noqa: E402
import evaluate_soft_cascade as cascade  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--track-index", type=Path, required=True)
    parser.add_argument("--tr-checkpoint", type=Path, required=True)
    parser.add_argument("--rd-checkpoint", type=Path, required=True)
    parser.add_argument("--rd-cache", type=Path, default=None)
    parser.add_argument("--calibration-partition", choices=["val"], default="val")
    parser.add_argument("--batch-size-tr", type=int, default=8)
    parser.add_argument("--batch-size-rd", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gate-epochs", type=int, default=200)
    parser.add_argument("--gate-learning-rate", type=float, default=0.01)
    parser.add_argument("--gate-weight-decay", type=float, default=0.0001)
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--initial-rd-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.gate_epochs < 1 or args.gate_learning_rate <= 0 or args.gate_hidden_dim < 1:
        raise ValueError("gate training parameters must be positive")
    if not 0 < args.initial_rd_weight < 1:
        raise ValueError("initial-rd-weight must be between zero and one")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    split_path = args.split.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.track_index = args.track_index.expanduser().resolve()
    args.tr_checkpoint = args.tr_checkpoint.expanduser().resolve()
    args.rd_checkpoint = args.rd_checkpoint.expanduser().resolve()
    args.rd_cache = args.rd_cache.expanduser().resolve() if args.rd_cache else None
    args.output_dir = output
    args.partition = args.calibration_partition
    device = torch.device(args.device)
    config = {
        "experiment_type": "fusion_gate_calibration",
        "experiment_label": "Frozen-checkpoint calibration quality gate",
        "dataset_root": str(args.dataset_root), "grouped_split": str(split_path),
        "grouped_split_sha256": sha256(split_path), "track_index": str(args.track_index),
        "tr_checkpoint": str(args.tr_checkpoint), "rd_checkpoint": str(args.rd_checkpoint),
        "rd_cache": str(args.rd_cache) if args.rd_cache else "",
        "calibration_partition": args.calibration_partition,
        "batch_size_tr": args.batch_size_tr, "batch_size_rd": args.batch_size_rd,
        "workers": args.workers, "gate_epochs": args.gate_epochs,
        "gate_learning_rate": args.gate_learning_rate, "gate_weight_decay": args.gate_weight_decay,
        "gate_hidden_dim": args.gate_hidden_dim, "initial_rd_weight": args.initial_rd_weight,
        "seed": args.seed, "device": str(device),
        "branches_frozen": True, "oof_used": False,
        "test_deferred": True,
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(output, {"phase": "trajectory_inference", "batch": 0, "total_batches": 0, "percent": 0.0})
    # Reuse the branch inference functions while translating their per-stage
    # batch progress into one smooth, honest overall progress bar.
    def calibration_progress(_output_dir: Path, payload: dict[str, object]) -> None:
        row = dict(payload)
        total = max(int(row.get("total_batches") or 0), 1)
        fraction = min(max(float(row.get("batch") or 0) / total, 0.0), 1.0)
        if row.get("phase") == "trajectory_inference":
            row["percent"] = 33.0 * fraction
        elif row.get("phase") == "rd_inference":
            row["percent"] = 33.0 + 34.0 * fraction
        write_progress(output, row)
    cascade.write_progress = calibration_progress
    split = load_grouped_split(split_path)
    tr_logits, tr_labels = infer_tr(args, split, device)
    rd_logits, rd_labels, rd_cache, rd_config = infer_rd(args, split, device)
    expected_ids = sorted((key for key, value in split.items() if value == args.partition), key=int)
    if set(tr_logits) != set(expected_ids) or set(rd_logits) != set(expected_ids) or tr_labels != rd_labels:
        raise ValueError("frozen branch evidence does not exactly cover the calibration partition")
    tr_tensor = torch.tensor(np.stack([tr_logits[key] for key in expected_ids]), dtype=torch.float32)
    frame_logits, frame_to_track = [], []
    for index, trajectory_id in enumerate(expected_ids):
        values = rd_logits[trajectory_id]
        frame_logits.extend(values)
        frame_to_track.extend([index] * len(values))
    fixed = SoftCascadeFusion(mode="fixed", fixed_rd_weight=args.initial_rd_weight)
    evidence = fixed(tr_tensor, torch.tensor(np.stack(frame_logits), dtype=torch.float32),
                     torch.tensor(frame_to_track, dtype=torch.long))
    records = []
    for index, trajectory_id in enumerate(expected_ids):
        truth = tr_labels[trajectory_id]
        records.append({
            "trajectory_id": trajectory_id, "true_class": truth, "true_label": CLASS_NAMES[truth],
            "tr_probabilities": evidence.tr_probabilities[index].tolist(),
            "tr_prediction": int(evidence.tr_predictions[index]),
            "rd_probabilities": evidence.rd_probabilities[index].tolist(),
            "rd_prediction": int(evidence.rd_predictions[index]),
            "rd_available": bool(evidence.rd_available[index]),
            "rd_frame_count": int(evidence.rd_frame_count[index]),
            "rd_consistency": float(evidence.rd_consistency[index]),
        })
    with (output / "calibration_scores.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "score_origin": "frozen_checkpoint_calibration", "source_partition": args.partition,
        "split_sha256": sha256(split_path), "grouped_split": str(split_path),
        "tr_checkpoint": str(args.tr_checkpoint), "rd_checkpoint": str(args.rd_checkpoint),
        "tr_checkpoint_sha256": sha256(args.tr_checkpoint),
        "rd_checkpoint_sha256": sha256(args.rd_checkpoint),
        "trajectory_count": len(records), "branches_frozen": True, "test_untouched": True,
    }
    (output / "calibration_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(output, {"phase": "gate_training", "batch": 0, "total_batches": 0, "epoch": 0,
                            "total_epochs": args.gate_epochs, "percent": 67.0})
    payload, records = train_gate_inline(
        records, epochs=args.gate_epochs, learning_rate=args.gate_learning_rate,
        weight_decay=args.gate_weight_decay, hidden_dim=args.gate_hidden_dim,
        initial_rd_weight=args.initial_rd_weight, seed=args.seed, device=device,
        output=output / "fusion_gate.pt", metadata=metadata,
        progress=lambda epoch, total, loss, accuracy: write_progress(
            output, {"phase": "gate_training", "epoch": epoch, "total_epochs": total,
                     "gate_loss": loss, "gate_accuracy": accuracy,
                     "percent": 67.0 + 33.0 * epoch / max(total, 1)}),
    )
    provenance = {**metadata, "partition": args.partition, "metrics_scope": "calibration fit; not an unbiased validation result",
                  "rd_cache": str(rd_cache.cache_dir) if rd_cache.cache_dir else None,
                  "rd_cache_hits": rd_cache.hits, "rd_cache_misses": rd_cache.misses,
                  "rd_configuration": {key: rd_config[key] for key in ("velocity_min", "velocity_max", "target_width", "resampling", "normalization", "input_mode", "model_head")}}
    summary = save_fusion_report(output, records, provenance)
    gate_metrics = {"metrics_scope": "calibration gate fit diagnostics; test untouched", "trajectory_count": len(records),
                    "tr_branch": classification_metrics(records, "tr_prediction"),
                    "rd_branch": classification_metrics(records, "rd_prediction"),
                    "gate_fit": classification_metrics(records, "fused_prediction"), "fit_summary": payload["fit_summary"]}
    (output / "gate_metrics.json").write_text(json.dumps(gate_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "gate_history.json").write_text(json.dumps(payload["history"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "ablation_complete.json").write_text(json.dumps({"gate_checkpoint": str(output / "fusion_gate.pt"), "calibration_scores": str(output / "calibration_scores.jsonl"), "test_deferred": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(output, {"phase": "complete", "percent": 100.0})
    print(json.dumps({"gate_checkpoint": str(output / "fusion_gate.pt"), "calibration_trajectory_count": len(records),
                      "gate_fit_macro_f1": summary["soft_cascade"]["macro_f1"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

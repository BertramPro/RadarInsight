"""Fit the quality-aware class gate from out-of-fold branch scores.

The input is intentionally a score artifact, not raw validation output.  A
sidecar metadata JSON must declare ``score_origin: OOF`` and
``source_partition: train``.  This prevents fitting the gate on scores made by
the same samples used to train either base branch, or on the held-out split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_fusion.model import (  # noqa: E402
    AggregatedRDEvidence,
    CLASS_NAMES,
    QualityAwareClassGate,
    SoftCascadeFusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-jsonl", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
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


def load_oof(path: Path, metadata: dict[str, object]) -> dict[str, torch.Tensor | list[str]]:
    origin = str(metadata.get("score_origin", "")).upper()
    partition = str(metadata.get("source_partition", "")).lower()
    if origin != "OOF" or partition != "train":
        raise ValueError(
            "gate fitting requires metadata score_origin=OOF and source_partition=train; "
            "validation/test or in-sample scores are rejected"
        )
    required = {"split_sha256", "tr_checkpoint", "rd_checkpoint"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"OOF metadata lacks required provenance fields: {missing}")

    ids: list[str] = []
    labels: list[int] = []
    tr: list[list[float]] = []
    rd: list[list[float]] = []
    available: list[bool] = []
    frame_count: list[float] = []
    consistency: list[float] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            trajectory_id = str(record["trajectory_id"])
            if trajectory_id in seen:
                raise ValueError(f"duplicate trajectory_id at line {line_number}: {trajectory_id}")
            seen.add(trajectory_id)
            label = int(record["true_class"])
            if not 0 <= label < len(CLASS_NAMES):
                raise ValueError(f"invalid true_class at line {line_number}: {label}")
            tr_prob = np.asarray(record["tr_probabilities"], dtype=np.float32)
            rd_prob = np.asarray(record["rd_probabilities"], dtype=np.float32)
            if tr_prob.shape != (5,) or rd_prob.shape != (5,):
                raise ValueError(f"probabilities must have five classes at line {line_number}")
            if not np.isfinite(tr_prob).all() or not np.isfinite(rd_prob).all():
                raise ValueError(f"non-finite probabilities at line {line_number}")
            if (tr_prob < 0).any() or (rd_prob < 0).any() or tr_prob.sum() <= 0 or rd_prob.sum() <= 0:
                raise ValueError(f"invalid probabilities at line {line_number}")
            ids.append(trajectory_id)
            labels.append(label)
            tr.append((tr_prob / tr_prob.sum()).tolist())
            rd.append((rd_prob / rd_prob.sum()).tolist())
            available.append(bool(record.get("rd_available", True)))
            frame_count.append(float(record.get("rd_frame_count", 1.0)))
            consistency.append(float(record.get("rd_consistency", 1.0)))
    if not ids:
        raise ValueError("OOF JSONL contains no records")
    return {
        "trajectory_ids": ids,
        "labels": torch.tensor(labels, dtype=torch.long),
        "tr_probabilities": torch.tensor(tr, dtype=torch.float32),
        "rd_probabilities": torch.tensor(rd, dtype=torch.float32),
        "available": torch.tensor(available, dtype=torch.bool),
        "frame_count": torch.tensor(frame_count, dtype=torch.float32),
        "consistency": torch.tensor(consistency, dtype=torch.float32).clamp(0.0, 1.0),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs and learning-rate must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    data = load_oof(args.oof_jsonl, metadata)
    device = torch.device(args.device)
    tr = data["tr_probabilities"].to(device)
    rd_prob = data["rd_probabilities"].to(device)
    labels = data["labels"].to(device)
    available = data["available"].to(device)
    frame_count = data["frame_count"].to(device)
    consistency = data["consistency"].to(device)
    frame_logits = rd_prob.clamp_min(1e-8).log()
    frame_to_track = torch.arange(len(labels), device=device, dtype=torch.long)
    model = SoftCascadeFusion(
        mode="quality_classwise",
        fixed_rd_weight=args.initial_rd_weight,
        hidden_dim=args.hidden_dim,
    ).to(device)
    # The gate is the only trainable object; branch scores are fixed OOF data.
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    model.train()
    rd = AggregatedRDEvidence(
        frame_logits=frame_logits,
        probabilities=rd_prob,
        predictions=rd_prob.argmax(dim=-1),
        available=available,
        frame_count=frame_count,
        consistency=consistency,
    )
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        fused, weights = model.fuse_probabilities(tr, rd)
        loss = F.nll_loss(fused.clamp_min(1e-8).log(), labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracy = float((fused.argmax(dim=-1) == labels).float().mean())
            history.append({
                "epoch": epoch,
                "loss": float(loss),
                "accuracy": accuracy,
                "mean_rd_weight": float(weights[available].mean()) if bool(available.any()) else 0.0,
            })
    model.eval()
    with torch.no_grad():
        fused, weights = model.fuse_probabilities(tr, rd)
        final_loss = float(F.nll_loss(fused.clamp_min(1e-8).log(), labels))
        final_accuracy = float((fused.argmax(dim=-1) == labels).float().mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fusion_state": model.state_dict(),
        "config": {
            "mode": "quality_classwise",
            "hidden_dim": args.hidden_dim,
            "initial_rd_weight": args.initial_rd_weight,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "class_names": CLASS_NAMES,
        },
        "provenance": {
            **metadata,
            "score_origin": "OOF",
            "fit_partition": "train",
            "oof_jsonl": str(args.oof_jsonl.resolve()),
            "oof_jsonl_sha256": sha256(args.oof_jsonl),
            "trajectory_count": len(labels),
        },
        "fit_summary": {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "mean_rd_weight": float(weights[available].mean()) if bool(available.any()) else 0.0,
        },
        "history": history,
    }
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output.resolve()), **payload["fit_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Generate strict train-partition OOF branch scores and fit the fusion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_fusion.data import (  # noqa: E402
    TrajectoryDataset,
    collate_trajectories,
    load_grouped_split,
    load_trajectory_records,
)
from radar_fusion.model import (  # noqa: E402
    AggregatedRDEvidence,
    B01_LABELS,
    CLASS_NAMES,
    SoftCascadeFusion,
    TrajectoryBranch,
    aggregate_rd_evidence,
    load_b01_trajectory_branch,
    load_checkpoint_metadata,
)
from radar_fusion.reporting import classification_metrics  # noqa: E402
from radar_rd.train import (  # noqa: E402
    RDCache,
    RDDataset,
    SmallRDCNN,
    build_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--track-index", type=Path, required=True)
    parser.add_argument("--tr-template-checkpoint", type=Path, required=True)
    parser.add_argument("--rd-template-checkpoint", type=Path, required=True)
    parser.add_argument("--rd-cache", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--tr-epochs", type=int, default=0,
                        help="Zero reuses the selected epoch stored in the TR template checkpoint")
    parser.add_argument("--rd-epochs", type=int, default=0,
                        help="Zero reuses the selected epoch stored in the RD template checkpoint")
    parser.add_argument("--batch-size-tr", type=int, default=8)
    parser.add_argument("--batch-size-rd", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gate-epochs", type=int, default=200)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-2)
    parser.add_argument("--gate-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-hidden-dim", type=int, default=32)
    parser.add_argument("--initial-rd-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_epoch(checkpoint: dict[str, Any], requested: int, branch: str) -> int:
    epoch = requested or int(checkpoint.get("best_epoch") or checkpoint.get("epoch") or 0)
    if epoch <= 0:
        raise ValueError(f"{branch} template checkpoint does not provide a usable selected epoch")
    return epoch


def fold_split(
    ids: np.ndarray,
    labels: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
    inner_val_fraction: float,
    seed: int,
) -> dict[str, str]:
    outer_train_ids = ids[outer_train_indices]
    outer_train_labels = labels[outer_train_indices]
    train_ids, val_ids = train_test_split(
        outer_train_ids,
        test_size=inner_val_fraction,
        random_state=seed,
        stratify=outer_train_labels,
    )
    result = {str(value): "train" for value in train_ids}
    result.update({str(value): "val" for value in val_ids})
    result.update({str(ids[index]): "test" for index in outer_test_indices})
    return result


def map_progress(stage_index: int, total_stages: int, stage_fraction: float) -> float:
    return 100.0 * (stage_index + min(max(stage_fraction, 0.0), 1.0)) / max(total_stages, 1)


def publish(
    output: Path,
    *,
    phase: str,
    stage_index: int,
    total_stages: int,
    stage_fraction: float = 0.0,
    fold: int = 0,
    folds: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "phase": phase,
        "stage": phase,
        "fold": fold,
        "total_folds": folds,
        "stage_index": stage_index + 1,
        "total_stages": total_stages,
        "stage_percent": 100.0 * min(max(stage_fraction, 0.0), 1.0),
        "percent": map_progress(stage_index, total_stages, stage_fraction),
    }
    if extra:
        payload.update(extra)
    write_json(output / "progress.json", payload)


def tail(path: Path, limit: int = 4000) -> str:
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
        return value[-limit:]
    except OSError:
        return ""


def run_child(
    command: list[str],
    *,
    cwd: Path,
    child_output: Path,
    stdout_path: Path,
    stderr_path: Path,
    parent_output: Path,
    phase: str,
    stage_index: int,
    total_stages: int,
    fold: int,
    folds: int,
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(cwd), stdout=stdout, stderr=stderr)
    while process.poll() is None:
        child = read_json(child_output / "progress.json", {}) or {}
        child_percent = float(child.get("percent", 0.0)) / 100.0
        publish(
            parent_output,
            phase=phase,
            stage_index=stage_index,
            total_stages=total_stages,
            stage_fraction=child_percent,
            fold=fold,
            folds=folds,
            extra={
                "epoch": child.get("epoch", 0),
                "total_epochs": child.get("total_epochs", 0),
                "batch": child.get("batch", 0),
                "total_batches": child.get("total_batches", 0),
                "child_phase": child.get("phase"),
            },
        )
        time.sleep(0.5)
    if process.returncode != 0:
        raise RuntimeError(f"{phase} failed with exit code {process.returncode}: {tail(stderr_path)}")
    publish(parent_output, phase=phase, stage_index=stage_index, total_stages=total_stages,
            stage_fraction=1.0, fold=fold, folds=folds)


@torch.inference_mode()
def infer_tr(
    checkpoint: Path,
    split_path: Path,
    track_index: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    progress: Callable[[int, int], None],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    split = load_grouped_split(split_path, allow_subset=True)
    records = load_trajectory_records(track_index, split, "test")
    loader = DataLoader(
        TrajectoryDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_trajectories,
        pin_memory=device.type == "cuda",
    )
    model = load_b01_trajectory_branch(checkpoint).to(device).eval()
    probabilities: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for batch_index, batch in enumerate(loader, start=1):
        logits = model(
            batch.sequence.to(device, non_blocking=True),
            batch.physical.to(device, non_blocking=True),
            batch.padding_mask.to(device, non_blocking=True),
        )
        values = torch.softmax(logits, dim=-1).cpu().numpy()
        for trajectory_id, label, probability in zip(batch.trajectory_ids, batch.labels.tolist(), values):
            probabilities[str(trajectory_id)] = probability
            labels[str(trajectory_id)] = int(label)
        progress(batch_index, len(loader))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, labels


@torch.inference_mode()
def infer_rd(
    checkpoint_path: Path,
    split_path: Path,
    dataset_root: Path,
    rd_cache_path: Path | None,
    batch_size: int,
    workers: int,
    device: torch.device,
    progress: Callable[[int, int], None],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    split = load_grouped_split(split_path, allow_subset=True)
    checkpoint = load_checkpoint_metadata(checkpoint_path)
    config = read_json(checkpoint_path.parent / "config.json", {}) or {}
    frames = [frame for frame in build_manifest(dataset_root) if split.get(frame.trajectory_id) == "test"]
    cache = None
    if rd_cache_path is not None:
        cache = RDCache(
            rd_cache_path,
            frames,
            velocity_min=float(config["velocity_min"]),
            velocity_max=float(config["velocity_max"]),
            target_width=int(config["target_width"]),
            resampling=str(config["resampling"]),
            allow_missing=True,
        )
    dataset = RDDataset(
        frames,
        float(checkpoint["mean"]),
        float(checkpoint["std"]),
        velocity_min=float(config["velocity_min"]),
        velocity_max=float(config["velocity_max"]),
        target_width=int(config["target_width"]),
        resampling=str(config["resampling"]),
        normalization=str(config["normalization"]),
        input_mode=str(config["input_mode"]),
        augmentation="off",
        rd_cache=cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    channels = 1 if config["input_mode"] == "rd" else 2
    model = SmallRDCNN(input_channels=channels, head=str(config.get("model_head", "global")))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    logits_by_id: dict[str, list[np.ndarray]] = {}
    labels: dict[str, int] = {}
    for batch_index, (images, targets, trajectory_ids) in enumerate(loader, start=1):
        logits = model(images.to(device, non_blocking=True)).cpu().numpy()
        for trajectory_id, label, values in zip(trajectory_ids, targets.tolist(), logits):
            logits_by_id.setdefault(str(trajectory_id), []).append(values)
            labels[str(trajectory_id)] = int(label)
        progress(batch_index, len(loader))
    ordered_ids = sorted(logits_by_id, key=int)
    frame_logits: list[np.ndarray] = []
    frame_to_track: list[int] = []
    for track_index, trajectory_id in enumerate(ordered_ids):
        values = logits_by_id[trajectory_id]
        frame_logits.extend(values)
        frame_to_track.extend([track_index] * len(values))
    evidence = aggregate_rd_evidence(
        torch.tensor(np.stack(frame_logits), dtype=torch.float32),
        torch.tensor(frame_to_track, dtype=torch.long),
        len(ordered_ids),
    )
    result = {
        trajectory_id: {
            "probabilities": evidence.probabilities[index].numpy(),
            "frame_count": int(evidence.frame_count[index]),
            "consistency": float(evidence.consistency[index]),
            "available": bool(evidence.available[index]),
        }
        for index, trajectory_id in enumerate(ordered_ids)
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, labels


def train_gate_inline(
    records: list[dict[str, Any]],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    initial_rd_weight: float,
    seed: int,
    device: torch.device,
    output: Path,
    metadata: dict[str, Any],
    progress: Callable[[int, int, float, float], None],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(seed)
    labels = torch.tensor([int(record["true_class"]) for record in records], dtype=torch.long, device=device)
    tr = torch.tensor([record["tr_probabilities"] for record in records], dtype=torch.float32, device=device)
    rd_prob = torch.tensor([record["rd_probabilities"] for record in records], dtype=torch.float32, device=device)
    available = torch.tensor([record["rd_available"] for record in records], dtype=torch.bool, device=device)
    frame_count = torch.tensor([record["rd_frame_count"] for record in records], dtype=torch.float32, device=device)
    consistency = torch.tensor([record["rd_consistency"] for record in records], dtype=torch.float32, device=device)
    rd = AggregatedRDEvidence(
        frame_logits=rd_prob.clamp_min(1e-8).log(),
        probabilities=rd_prob,
        predictions=rd_prob.argmax(dim=-1),
        available=available,
        frame_count=frame_count,
        consistency=consistency,
    )
    model = SoftCascadeFusion(
        mode="quality_classwise",
        fixed_rd_weight=initial_rd_weight,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        fused, weights = model.fuse_probabilities(tr, rd)
        loss = F.nll_loss(fused.clamp_min(1e-8).log(), labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracy = float((fused.argmax(dim=-1) == labels).float().mean())
            mean_weight = float(weights[available].mean()) if bool(available.any()) else 0.0
        row = {"epoch": epoch, "loss": float(loss.detach()), "accuracy": accuracy,
               "mean_rd_weight": mean_weight}
        history.append(row)
        progress(epoch, epochs, row["loss"], accuracy)
    model.eval()
    with torch.no_grad():
        fused, weights = model.fuse_probabilities(tr, rd)
        predictions = fused.argmax(dim=-1).cpu().tolist()
        final_loss = float(F.nll_loss(fused.clamp_min(1e-8).log(), labels))
    for index, record in enumerate(records):
        record["rd_class_weights"] = weights[index].cpu().tolist()
        record["fused_probabilities"] = fused[index].cpu().tolist()
        record["fused_prediction"] = int(predictions[index])
    payload = {
        "fusion_state": model.state_dict(),
        "config": {
            "mode": "quality_classwise",
            "hidden_dim": hidden_dim,
            "initial_rd_weight": initial_rd_weight,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "class_names": CLASS_NAMES,
        },
        "provenance": metadata,
        "fit_summary": {
            "loss": final_loss,
            "accuracy": float((fused.argmax(dim=-1) == labels).float().mean()),
            "mean_rd_weight": float(weights[available].mean()) if bool(available.any()) else 0.0,
            "class_mean_rd_weights": weights.mean(dim=0).cpu().tolist(),
        },
        "history": history,
    }
    torch.save(payload, output)
    return payload, records


def main() -> None:
    args = parse_args()
    if not 2 <= args.folds <= 10:
        raise ValueError("folds must be between 2 and 10")
    if not 0.05 <= args.inner_val_fraction <= 0.40:
        raise ValueError("inner-val-fraction must be between 0.05 and 0.40")
    if args.gate_epochs <= 0 or args.gate_learning_rate <= 0 or args.gate_hidden_dim <= 0:
        raise ValueError("gate training parameters must be positive")
    if not 0.0 < args.initial_rd_weight < 1.0:
        raise ValueError("initial-rd-weight must be between zero and one")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    device = torch.device(args.device)
    split_path = args.split.expanduser().resolve()
    track_index = args.track_index.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    tr_template = args.tr_template_checkpoint.expanduser().resolve()
    rd_template = args.rd_template_checkpoint.expanduser().resolve()
    rd_cache_path = args.rd_cache.expanduser().resolve() if args.rd_cache else None
    source_split = load_grouped_split(split_path)
    train_records = load_trajectory_records(track_index, source_split, "train")
    ids = np.asarray([record.trajectory_id for record in train_records])
    labels = np.asarray([record.label for record in train_records], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if int(class_counts.min()) < args.folds:
        raise ValueError(f"Each class needs at least {args.folds} train trajectories; counts={class_counts.tolist()}")
    tr_template_data = load_checkpoint_metadata(tr_template)
    rd_template_data = load_checkpoint_metadata(rd_template)
    tr_epochs = source_epoch(tr_template_data, args.tr_epochs, "TR")
    rd_epochs = source_epoch(rd_template_data, args.rd_epochs, "RD")
    tr_config = tr_template_data.get("config") if isinstance(tr_template_data.get("config"), dict) else {}
    rd_config = read_json(rd_template.parent / "config.json", {}) or {}
    augmentation = rd_config.get("augmentation", "off")
    if isinstance(augmentation, dict):
        augmentation = augmentation.get("mode", "off")
    config = {
        "experiment_type": "fusion_gate_training",
        "experiment_label": "Strict train-partition OOF quality-aware class gate",
        "dataset_root": str(dataset_root),
        "grouped_split": str(split_path),
        "grouped_split_sha256": file_sha256(split_path),
        "track_index": str(track_index),
        "tr_checkpoint": str(tr_template),
        "rd_checkpoint": str(rd_template),
        "rd_cache": str(rd_cache_path) if rd_cache_path else "",
        "folds": args.folds,
        "inner_val_fraction": args.inner_val_fraction,
        "tr_epochs": tr_epochs,
        "rd_epochs": rd_epochs,
        "batch_size_tr": args.batch_size_tr,
        "batch_size_rd": args.batch_size_rd,
        "workers": args.workers,
        "gate_epochs": args.gate_epochs,
        "gate_learning_rate": args.gate_learning_rate,
        "gate_weight_decay": args.gate_weight_decay,
        "gate_hidden_dim": args.gate_hidden_dim,
        "initial_rd_weight": args.initial_rd_weight,
        "seed": args.seed,
        "device": str(device),
        "source_partition": "train",
        "validation_test_untouched": True,
    }
    write_json(output / "config.json", config)
    total_stages = args.folds * 4 + 1
    publish(output, phase="oof_preparing", stage_index=0, total_stages=total_stages,
            folds=args.folds, extra={"trajectory_count": len(ids)})
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_root = output / "folds"
    log_root = output / "fold_logs"
    oof_records: list[dict[str, Any]] = []
    fold_manifest: list[dict[str, Any]] = []
    python = sys.executable
    stage_index = 0
    for fold_zero, (outer_train, outer_test) in enumerate(splitter.split(ids, labels)):
        fold_number = fold_zero + 1
        split = fold_split(ids, labels, outer_train, outer_test, args.inner_val_fraction, args.seed + fold_number)
        fold_dir = fold_root / f"fold_{fold_number:02d}"
        split_file = fold_dir / "split.json"
        write_json(split_file, split)
        counts = {name: sum(value == name for value in split.values()) for name in ("train", "val", "test")}
        fold_manifest.append({"fold": fold_number, "split": str(split_file), **counts})
        tr_output = fold_dir / "tr"
        tr_command = [
            python, str(PROJECT_ROOT / "scripts" / "train_tr_only.py"),
            "--output-dir", str(tr_output),
            "--split", str(split_file),
            "--track-index", str(track_index),
            "--epochs", str(tr_epochs),
            "--batch-size", str(args.batch_size_tr),
            "--workers", str(args.workers),
            "--learning-rate", str(tr_config.get("learning_rate", 2e-4)),
            "--weight-decay", str(tr_config.get("weight_decay", 1e-4)),
            "--patience", "0",
            "--seed", str(args.seed + fold_number),
            "--dropout", str(tr_config.get("dropout", 0.1)),
            "--unknown-sample-boost", str(tr_config.get("unknown_sample_boost", 1.0)),
            "--clutter-sample-boost", str(tr_config.get("clutter_sample_boost", 2.0)),
            "--skip-test",
            "--allow-split-subset",
            "--device", str(device),
        ]
        run_child(
            tr_command,
            cwd=PROJECT_ROOT,
            child_output=tr_output,
            stdout_path=log_root / f"fold_{fold_number:02d}_tr.log",
            stderr_path=log_root / f"fold_{fold_number:02d}_tr.err.log",
            parent_output=output,
            phase="oof_tr_training",
            stage_index=stage_index,
            total_stages=total_stages,
            fold=fold_number,
            folds=args.folds,
        )
        stage_index += 1
        tr_probabilities, tr_labels = infer_tr(
            tr_output / "best.pt", split_file, track_index, args.batch_size_tr, args.workers, device,
            lambda batch, total, si=stage_index, fn=fold_number: publish(
                output, phase="oof_tr_inference", stage_index=si, total_stages=total_stages,
                stage_fraction=batch / max(total, 1), fold=fn, folds=args.folds,
                extra={"batch": batch, "total_batches": total},
            ),
        )
        stage_index += 1
        rd_output = fold_dir / "rd"
        rd_command = [
            python, str(PROJECT_ROOT / "radar_rd" / "train.py"),
            "--dataset-root", str(dataset_root),
            "--output-dir", str(rd_output),
            "--grouped-split", str(split_file),
            "--epochs", str(rd_epochs),
            "--batch-size", str(args.batch_size_rd),
            "--workers", str(args.workers),
            "--max-train-frames-per-trajectory", str(rd_config.get("max_train_frames_per_trajectory", 32)),
            "--norm-samples", str(rd_config.get("norm_samples", 2048)),
            "--learning-rate", str(rd_config.get("learning_rate", 3e-4)),
            "--weight-decay", str(rd_config.get("weight_decay", 1e-4)),
            "--patience", str(max(1, int(rd_config.get("patience", 10)))),
            "--seed", str(args.seed + fold_number),
            "--velocity-min", str(rd_config.get("velocity_min", -90.0)),
            "--velocity-max", str(rd_config.get("velocity_max", 89.0)),
            "--target-width", str(rd_config.get("target_width", 900)),
            "--resampling", str(rd_config.get("resampling", "db_linear")),
            "--normalization", str(rd_config.get("normalization", "global_z")),
            "--input-mode", str(rd_config.get("input_mode", "rd")),
            "--model-head", str(rd_config.get("model_head", "global")),
            "--augmentation", str(augmentation),
            "--skip-test",
        ]
        # Each OOF fold has its own train/validation identity.  A cache made
        # from the outer fixed split must never be presented to train.py here:
        # train.py correctly rejects its different cache_identity before it
        # could fall back on missing frames.  Fold training therefore uses
        # deterministic on-demand preprocessing.  The following fold-test
        # inference can still use the cache opportunistically (with misses
        # falling back safely in infer_rd()).
        run_child(
            rd_command,
            cwd=PROJECT_ROOT,
            child_output=rd_output,
            stdout_path=log_root / f"fold_{fold_number:02d}_rd.log",
            stderr_path=log_root / f"fold_{fold_number:02d}_rd.err.log",
            parent_output=output,
            phase="oof_rd_training",
            stage_index=stage_index,
            total_stages=total_stages,
            fold=fold_number,
            folds=args.folds,
        )
        stage_index += 1
        rd_evidence, rd_labels = infer_rd(
            rd_output / "best.pt", split_file, dataset_root, rd_cache_path,
            args.batch_size_rd, args.workers, device,
            lambda batch, total, si=stage_index, fn=fold_number: publish(
                output, phase="oof_rd_inference", stage_index=si, total_stages=total_stages,
                stage_fraction=batch / max(total, 1), fold=fn, folds=args.folds,
                extra={"batch": batch, "total_batches": total},
            ),
        )
        stage_index += 1
        expected = sorted((trajectory_id for trajectory_id, value in split.items() if value == "test"), key=int)
        if set(tr_probabilities) != set(expected) or set(rd_evidence) != set(expected):
            raise ValueError(f"Fold {fold_number} inference does not exactly cover its outer holdout")
        for trajectory_id in expected:
            if tr_labels[trajectory_id] != rd_labels[trajectory_id]:
                raise ValueError(f"TR/RD label mismatch for trajectory {trajectory_id}")
            tr_probability = tr_probabilities[trajectory_id]
            rd_item = rd_evidence[trajectory_id]
            oof_records.append({
                "trajectory_id": trajectory_id,
                "outer_fold": fold_number,
                "true_class": tr_labels[trajectory_id],
                "true_label": B01_LABELS[tr_labels[trajectory_id]],
                "tr_probabilities": tr_probability.tolist(),
                "tr_prediction": int(tr_probability.argmax()),
                "rd_probabilities": rd_item["probabilities"].tolist(),
                "rd_prediction": int(rd_item["probabilities"].argmax()),
                "rd_available": rd_item["available"],
                "rd_frame_count": rd_item["frame_count"],
                "rd_consistency": rd_item["consistency"],
            })
    oof_records.sort(key=lambda item: int(item["trajectory_id"]))
    if {record["trajectory_id"] for record in oof_records} != set(ids.tolist()):
        raise ValueError("OOF records do not cover every source training trajectory exactly once")
    if len(oof_records) != len(ids):
        raise ValueError("OOF records contain duplicate trajectories")
    write_json(output / "fold_manifest.json", fold_manifest)
    with (output / "oof_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in oof_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "score_origin": "OOF",
        "source_partition": "train",
        "split_sha256": file_sha256(split_path),
        "grouped_split": str(split_path),
        "tr_checkpoint": str(tr_template),
        "rd_checkpoint": str(rd_template),
        "folds": args.folds,
        "inner_val_fraction": args.inner_val_fraction,
        "trajectory_count": len(oof_records),
        "validation_test_untouched": True,
        "fold_models_randomly_initialized": True,
        "fold_epoch_source": "template_selected_epoch" if args.tr_epochs == 0 and args.rd_epochs == 0 else "explicit_or_template",
    }
    write_json(output / "oof_metadata.json", metadata)
    gate_checkpoint = output / "fusion_gate.pt"
    payload, gate_records = train_gate_inline(
        oof_records,
        epochs=args.gate_epochs,
        learning_rate=args.gate_learning_rate,
        weight_decay=args.gate_weight_decay,
        hidden_dim=args.gate_hidden_dim,
        initial_rd_weight=args.initial_rd_weight,
        seed=args.seed,
        device=device,
        output=gate_checkpoint,
        metadata=metadata,
        progress=lambda epoch, total, loss, accuracy: publish(
            output, phase="gate_training", stage_index=stage_index, total_stages=total_stages,
            stage_fraction=epoch / max(total, 1), folds=args.folds,
            extra={"epoch": epoch, "total_epochs": total, "gate_loss": loss, "gate_accuracy": accuracy},
        ),
    )
    with (output / "oof_gate_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for record in gate_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = {
        "metrics_scope": "OOF gate-fit diagnostics; not validation or test performance",
        "trajectory_count": len(gate_records),
        "tr_branch": classification_metrics(gate_records, "tr_prediction"),
        "rd_branch": classification_metrics(gate_records, "rd_prediction"),
        "gate_fit": classification_metrics(gate_records, "fused_prediction"),
        "fit_summary": payload["fit_summary"],
    }
    write_json(output / "gate_metrics.json", metrics)
    write_json(output / "gate_history.json", payload["history"])
    write_json(output / "ablation_complete.json", {
        "gate_checkpoint": str(gate_checkpoint),
        "oof_trajectory_count": len(gate_records),
        "folds": args.folds,
        "test_deferred": True,
    })
    publish(output, phase="complete", stage_index=total_stages - 1, total_stages=total_stages,
            stage_fraction=1.0, folds=args.folds,
            extra={"gate_checkpoint": str(gate_checkpoint), "oof_trajectory_count": len(gate_records)})
    print(json.dumps({
        "gate_checkpoint": str(gate_checkpoint),
        "oof_trajectory_count": len(gate_records),
        "tr_oof_macro_f1": metrics["tr_branch"]["macro_f1"],
        "rd_oof_macro_f1": metrics["rd_branch"]["macro_f1"],
        "gate_fit_macro_f1": metrics["gate_fit"]["macro_f1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

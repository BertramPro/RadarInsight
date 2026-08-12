"""Train the standalone B01-compatible trajectory branch.

The monitor uses this program for TR-only training.  Checkpoint evaluation is
kept in evaluate_tr_only.py and is deliberately a separate experiment type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_fusion.data import (  # noqa: E402
    TrajectoryDataset,
    collate_trajectories,
    load_grouped_split,
    load_trajectory_records,
)
from radar_fusion.model import B01_LABELS, CLASS_NAMES, TrajectoryBranch  # noqa: E402
from radar_fusion.reporting import classification_metrics  # noqa: E402
from radar_fusion.trajectory_augmentation import label_counts  # noqa: E402
from radar_fusion.partition_augmentation import expand_trajectory_records, validate_targets  # noqa: E402
from radar_fusion.trajectory_cache import DEFAULT_TRAJECTORY_CACHE_ROOT, load_or_build  # noqa: E402


DEFAULT_SPLIT = Path(r"K:\radar\main\data\manifests\cq08_grouped_split_f.json")
DEFAULT_TRACK_INDEX = Path(r"K:\radar\main\data\processed\expert1_track_index.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--track-index", type=Path, default=DEFAULT_TRACK_INDEX)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("cosine", "none"), default="none")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-cosface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unknown-sample-boost", type=float, default=1.0)
    parser.add_argument("--clutter-sample-boost", type=float, default=2.0)
    parser.add_argument("--bird-sample-boost", type=float, default=1.0)
    parser.add_argument("--drone-sample-boost", type=float, default=1.0)
    parser.add_argument("--balloon-sample-boost", type=float, default=1.0)
    parser.add_argument("--sampling-mode", choices=("b01_balanced", "inverse_frequency"), default="b01_balanced")
    parser.add_argument("--sampling-protocol", choices=("coverage_plus_boost", "strict_b01_replacement"), default="coverage_plus_boost")
    parser.add_argument("--manual-sampling-boosts-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augmentation-frame-drop-fraction", type=float, default=0.10)
    parser.add_argument("--augmentation-amplitude-scale-max-delta", type=float, default=0.05)
    parser.add_argument("--augmentation-snr-offset-db", type=float, default=1.0)
    parser.add_argument("--checkpoint-selection-metric", choices=("macro_f1", "bird_f1"), default="macro_f1")
    parser.add_argument("--class-weight-mode", choices=("inverse_sqrt", "class_balanced"), default="inverse_sqrt")
    parser.add_argument("--class-balanced-beta", type=float, default=0.999)
    parser.add_argument("--class-weight-floor", type=float, default=0.25)
    parser.add_argument("--class-weight-cap", type=float, default=2.0)
    parser.add_argument(
        "--manual-class-loss-weights", type=float, nargs=5, default=None,
        metavar="WEIGHT",
        help="Optional manual Cross-Entropy weights in label order: Drone, Bird, Balloon, Clutter, Unknown.",
    )
    parser.add_argument("--partition-augmentation-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition-augmentation-method", choices=("perturbation", "smote"), default="perturbation")
    parser.add_argument("--partition-augmentation-train-enabled", type=int, nargs=5, default=None,
                        metavar="ON", help="Per-class virtual-copy switches in label order (1/0).")
    parser.add_argument("--sampling-class-enabled", type=int, nargs=5, default=None,
                        metavar="ON", help="Per-class sampler weighting switches in label order (1/0).")
    parser.add_argument("--class-loss-enabled", type=int, nargs=5, default=None,
                        metavar="ON", help="Per-class loss weighting switches in label order (1/0).")
    parser.add_argument("--partition-augmentation-targets-train", type=int, nargs=5, default=None)
    parser.add_argument("--partition-augmentation-targets-val", type=int, nargs=5, default=None)
    parser.add_argument("--partition-augmentation-targets-test", type=int, nargs=5, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--allow-split-subset", action="store_true",
                        help="Allow an explicit grouped split containing a strict CQ-08 subset")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
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


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


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


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu() if isinstance(state["torch"], torch.Tensor) else state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = [value.detach().cpu() if isinstance(value, torch.Tensor) else value for value in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def sampling_weights(
    records,
    *,
    sampling_mode: str = "b01_balanced",
    drone_boost: float = 1.0,
    bird_boost: float = 1.0,
    balloon_boost: float = 1.0,
    clutter_boost: float = 1.0,
    unknown_boost: float = 1.0,
    class_enabled: list[bool] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return auditable sampler weights for the selected protocol."""
    counts = np.bincount([record.label for record in records], minlength=5).astype(np.float64)
    if sampling_mode == "b01_balanced":
        base = np.power(float(max(counts.max(), 1.0)) / np.maximum(counts, 1.0), 0.25)
        base_rule = "B01 source-label balance: (max_class_count / class_count)^0.25"
    elif sampling_mode == "inverse_frequency":
        base = 1.0 / np.maximum(counts, 1.0)
        base_rule = "per-record 1 / expanded_class_count"
    else:
        raise ValueError(f"unsupported sampling mode: {sampling_mode}")
    multipliers = np.asarray(
        [drone_boost, bird_boost, balloon_boost, clutter_boost, unknown_boost],
        dtype=np.float64,
    )
    enabled = [True] * 5 if class_enabled is None else [bool(value) for value in class_enabled]
    if len(enabled) != 5:
        raise ValueError("sampling-class-enabled must contain five values")
    # A disabled class bypasses the sampler weighting stage entirely.
    weights = torch.as_tensor([
        base[record.label] * multipliers[record.label] if enabled[record.label] else 1.0
        for record in records
    ], dtype=torch.double)
    total = float(weights.sum().item())
    class_masses = [float(weights[[index for index, record in enumerate(records) if record.label == label]].sum().item()) for label in range(5)]
    return weights, {
        "sampling_mode": sampling_mode, "base_rule": base_rule,
        "expanded_class_counts": {str(index): int(value) for index, value in enumerate(counts)},
        "extra_sample_boost": {
            "drone": drone_boost, "bird": bird_boost, "balloon": balloon_boost,
            "clutter": clutter_boost, "unknown": unknown_boost,
        },
        "sampling_class_enabled": enabled,
        "class_sampling_probability": {str(index): mass / total if total else 0.0 for index, mass in enumerate(class_masses)},
    }


class CoveragePlusBoostSampler(Sampler[int]):
    """Visit every record once, then add a bounded weighted class over-sample."""

    def __init__(self, records, weights: torch.Tensor, *, seed: int, max_extra_ratio: float = 1.0) -> None:
        self.labels = [int(record.label) for record in records]
        self.weights = weights.detach().cpu().double()
        self.seed = int(seed)
        self.max_extra_ratio = float(max_extra_ratio)
        self.epoch = 0
        self.last_audit: dict[str, Any] = {}
        self._cached_epoch: int | None = None
        self._cached_indices: list[int] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_indices = None

    def __len__(self) -> int:
        indices, _ = self._build_for_epoch()
        return len(indices)

    def _build_for_epoch(self) -> tuple[list[int], dict[str, Any]]:
        if self._cached_epoch == self.epoch and self._cached_indices is not None:
            return self._cached_indices, self.last_audit
        n = len(self.labels)
        if n == 0:
            audit = {"protocol": "coverage_plus_boost", "epoch": self.epoch,
                     "coverage_count": 0, "extra_count": 0, "total_count": 0,
                     "unique_count": 0, "duplicate_sampling": False}
            self.last_audit = audit
            self._cached_epoch, self._cached_indices = self.epoch, []
            return [], audit
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        base_indices = torch.randperm(n, generator=generator).tolist()
        counts = np.bincount(self.labels, minlength=5).astype(np.float64)
        # All switches off is the explicit no-reweighting mode: no extra pass.
        weighted_active = any(abs(float(self.weights[index]) - 1.0) > 1e-12 for index in range(n))
        extra_by_class = np.zeros(5, dtype=np.int64)
        if weighted_active:
            masses = np.zeros(5, dtype=np.float64)
            for index, label in enumerate(self.labels):
                masses[label] += float(self.weights[index])
            target = masses / masses.sum() if masses.sum() else counts / max(counts.sum(), 1.0)
            raw = counts / max(counts.sum(), 1.0)
            desired = np.maximum(0.0, target * n - counts)
            extra_total = min(n, int(np.ceil(float(desired.sum()) * self.max_extra_ratio)))
            if extra_total > 0 and desired.sum() > 0:
                exact = desired / desired.sum() * extra_total
                extra_by_class = np.floor(exact).astype(np.int64)
                remainder = int(extra_total - int(extra_by_class.sum()))
                order = np.argsort(-(exact - extra_by_class), kind="stable")
                for label in order[:remainder]:
                    extra_by_class[int(label)] += 1
        extra_indices: list[int] = []
        for label, amount in enumerate(extra_by_class.tolist()):
            candidates = [index for index, value in enumerate(self.labels) if value == label]
            if not candidates or amount <= 0:
                continue
            candidate_weights = self.weights[candidates].clamp_min(1e-12)
            chosen = torch.multinomial(candidate_weights, amount, replacement=True, generator=generator).tolist()
            extra_indices.extend(candidates[index] for index in chosen)
        if extra_indices:
            order = torch.randperm(len(extra_indices), generator=generator).tolist()
            extra_indices = [extra_indices[index] for index in order]
        result = base_indices + extra_indices
        final_counts = np.bincount([self.labels[index] for index in result], minlength=5).astype(int)
        audit = {
            "protocol": "coverage_plus_boost", "epoch": self.epoch,
            "coverage_count": n, "extra_count": len(extra_indices), "total_count": len(result),
            "unique_count": len(set(result)), "duplicate_sampling": len(result) != len(set(result)),
            "coverage_class_counts": counts.astype(int).tolist(), "extra_class_counts": extra_by_class.tolist(),
            "final_class_counts": final_counts.tolist(),
            "coverage_class_probability": (counts / max(n, 1)).tolist(),
            "extra_class_probability": (extra_by_class / max(len(extra_indices), 1)).tolist(),
            "final_class_probability": (final_counts / max(len(result), 1)).tolist(),
            "max_extra_ratio": self.max_extra_ratio,
        }
        self.last_audit = audit
        self._cached_epoch, self._cached_indices = self.epoch, result
        return result, audit

    def __iter__(self):
        indices, _ = self._build_for_epoch()
        return iter(indices)


class AuditedReplacementSampler(Sampler[int]):
    """Audited equivalent of WeightedRandomSampler(replacement=True)."""

    def __init__(self, weights: torch.Tensor, *, labels: list[int] | None = None, num_samples: int, seed: int) -> None:
        self.weights = weights.detach().cpu().double()
        self.labels = list(labels or [0] * int(weights.numel()))
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self.last_audit: dict[str, Any] = {}
        self._cached_epoch: int | None = None
        self._cached_indices: list[int] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_indices = None

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        if self._cached_epoch == self.epoch and self._cached_indices is not None:
            return iter(self._cached_indices)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights.clamp_min(1e-12), self.num_samples, replacement=True, generator=generator).tolist()
        final_counts = np.bincount([self.labels[index] for index in indices], minlength=5).astype(int)
        self.last_audit = {"protocol": "strict_b01_replacement", "epoch": self.epoch,
                           "coverage_count": 0, "extra_count": self.num_samples,
                           "total_count": self.num_samples, "unique_count": len(set(indices)),
                           "duplicate_sampling": len(indices) != len(set(indices)),
                           "coverage_class_counts": [0] * 5, "extra_class_counts": final_counts.tolist(),
                           "final_class_counts": final_counts.tolist(),
                           "final_class_probability": (final_counts / max(self.num_samples, 1)).tolist()}
        self._cached_epoch, self._cached_indices = self.epoch, indices
        return iter(indices)


def make_loader(records, batch_size: int, workers: int, *, train: bool, seed: int,
                sampling_protocol: str = "coverage_plus_boost",
                sampling_mode: str = "b01_balanced",
                drone_boost: float = 1.0, bird_boost: float = 1.0,
                balloon_boost: float = 1.0, clutter_boost: float = 1.0,
                unknown_boost: float = 1.0, class_enabled: list[bool] | None = None,
                dataset=None) -> DataLoader:
    dataset = dataset if dataset is not None else TrajectoryDataset(records)
    kwargs = dict(batch_size=batch_size, num_workers=workers, collate_fn=collate_trajectories,
                  pin_memory=torch.cuda.is_available())
    if not train:
        return DataLoader(dataset, shuffle=False, **kwargs)
    if sampling_protocol not in {"coverage_plus_boost", "strict_b01_replacement"}:
        raise ValueError(f"unsupported sampling protocol: {sampling_protocol}")
    weights, _ = sampling_weights(
        records, sampling_mode=sampling_mode, drone_boost=drone_boost, bird_boost=bird_boost,
        balloon_boost=balloon_boost, clutter_boost=clutter_boost,
        unknown_boost=unknown_boost, class_enabled=class_enabled,
    )
    if sampling_protocol == "strict_b01_replacement":
        sampler = AuditedReplacementSampler(weights, labels=[int(record.label) for record in records], num_samples=len(records), seed=seed)
    else:
        sampler = CoveragePlusBoostSampler(records, weights, seed=seed)
    return DataLoader(dataset, sampler=sampler, **kwargs)


def fit_source_stats(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for index in range(len(dataset)):
        values.append(dataset[index].sequence)
    cat = torch.cat(values, dim=0)
    return cat.mean(dim=0), cat.std(dim=0).clamp_min(1e-3)


def select_class_weights(counts: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Build relative CE weights, then normalize and apply per-class switches."""
    if args.manual_class_loss_weights is not None:
        weights = torch.as_tensor(args.manual_class_loss_weights, dtype=torch.float32)
        if weights.numel() != 5 or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("manual class-loss weights must be five positive finite values")
    else:
        counts = counts.float().clamp_min(1.0)
        if args.class_weight_mode == "inverse_sqrt":
            weights = torch.rsqrt(counts)
        else:
            if not 0.0 <= args.class_balanced_beta < 1.0:
                raise ValueError("class-balanced-beta must be in [0, 1)")
            beta = torch.tensor(args.class_balanced_beta, dtype=torch.float32)
            weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts)).clamp_min(1e-12)
    weights = weights / weights.mean().clamp_min(1e-12)
    weights = weights.clamp(args.class_weight_floor, args.class_weight_cap)
    enabled = [True] * 5 if args.class_loss_enabled is None else [bool(value) for value in args.class_loss_enabled]
    if len(enabled) != 5:
        raise ValueError("class-loss-enabled must contain five values")
    weights = torch.where(torch.as_tensor(enabled, dtype=torch.bool), weights, torch.ones_like(weights))
    return weights


def forward_batch(model, batch, device, criterion, train: bool) -> tuple[torch.Tensor, float, int, int]:
    sequence = batch.sequence.to(device, non_blocking=True)
    physical = batch.physical.to(device, non_blocking=True)
    mask = batch.padding_mask.to(device, non_blocking=True)
    labels = batch.labels.to(device, non_blocking=True)
    logits = model(sequence, physical, mask, labels=labels if train else None)
    loss = criterion(logits, labels)
    correct = int((logits.argmax(dim=-1) == labels).sum().item())
    return logits, float(loss.detach().item()), correct, int(labels.numel())


@torch.inference_mode()
def evaluate(model, loader, device, criterion, output: Path, epoch: int, total_epochs: int, phase: str = "validation") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    loss_sum = 0.0
    count = 0
    records: list[dict[str, Any]] = []
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        logits, loss, _, batch_count = forward_batch(model, batch, device, criterion, False)
        loss_sum += loss * batch_count
        count += batch_count
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = logits.argmax(dim=-1).cpu().tolist()
        for trajectory_id, label, probs, prediction in zip(batch.trajectory_ids, batch.labels.tolist(), probabilities, predictions):
            records.append({
                "trajectory_id": trajectory_id,
                "true_class": int(label),
                "true_label": B01_LABELS[int(label)],
                "tr_logits": probs.tolist(),
                "tr_probabilities": probs.tolist(),
                "tr_prediction": int(prediction),
                "tr_prediction_label": B01_LABELS[int(prediction)],
                "prediction": int(prediction),
                "prediction_label": B01_LABELS[int(prediction)],
            })
        write_json(output / "progress.json", {
            "phase": phase, "epoch": epoch, "total_epochs": total_epochs,
            "batch": batch_index, "total_batches": total_batches,
            "percent": 100.0 * batch_index / max(total_batches, 1),
            "val_loss": loss_sum / max(count, 1),
        })
    metrics = classification_metrics(records, "prediction")
    metrics["loss"] = loss_sum / max(count, 1)
    metrics["epoch"] = epoch
    return metrics, records


def save_last(path: Path, model, optimizer, scheduler, epoch, best_epoch, best_f1, best_selection_score, stale, history, center, scale, config):
    torch.save({
        "kind": "tr_training_last",
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_selection_score": best_selection_score,
        "stale_epochs": stale,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "source_center": center,
        "source_scale": scale,
        "config": config,
        "rng_state": capture_rng(),
    }, path)


def checkpoint_for_model(path: Path, model, epoch: int, best_f1: float, best_selection_score: float, center, scale, config):
    state = model.state_dict()
    torch.save({
        "kind": "tr_training_best",
        "epoch": epoch,
        "best_epoch": epoch,
        "best_val_macro_f1": best_f1,
        "best_selection_score": best_selection_score,
        "model_state": state,
        "model_state_dict": state,
        "representation_builder_state_dict": model.representation_builder.state_dict(),
        "source_center": center,
        "source_scale": scale,
        "config": config,
        "labels": B01_LABELS,
        "source_feature_stats": {"cq08_track": {"center": center, "scale": scale}},
    }, path)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    write_json(output / "progress.json", {"phase": "preparing", "stage": "loading_fixed_split", "epoch": 0, "total_epochs": args.epochs, "batch": 0, "total_batches": 0, "percent": 0.0})
    split = load_grouped_split(args.split, allow_subset=args.allow_split_subset)
    if not 0.0 <= args.augmentation_frame_drop_fraction <= 0.40:
        raise ValueError("augmentation-frame-drop-fraction must be in [0, 0.40]")
    if args.augmentation_amplitude_scale_max_delta < 0.0 or args.augmentation_snr_offset_db < 0.0:
        raise ValueError("trajectory augmentation magnitudes must be non-negative")
    if args.class_weight_floor < 0.0 or args.class_weight_cap <= 0.0 or args.class_weight_floor > args.class_weight_cap:
        raise ValueError("invalid automatic class-loss weight floor/cap")
    if args.manual_class_loss_weights is not None and (
        len(args.manual_class_loss_weights) != 5
        or any(not 0.0 < value <= 20.0 for value in args.manual_class_loss_weights)
    ):
        raise ValueError("manual-class-loss-weights must contain five values in (0, 20]")
    if any(value <= 0.0 for value in (
        args.drone_sample_boost, args.bird_sample_boost, args.balloon_sample_boost,
        args.clutter_sample_boost, args.unknown_sample_boost,
    )):
        raise ValueError("all sample boosts must be positive")
    base_train_records = load_trajectory_records(args.track_index, split, "train")
    train_records = list(base_train_records)
    val_records = load_trajectory_records(args.track_index, split, "val")
    test_records = load_trajectory_records(args.track_index, split, "test") if not args.skip_test else []
    train_targets = validate_targets(args.partition_augmentation_targets_train, train_records)
    train_augmentation_enabled = [True] * 5 if args.partition_augmentation_train_enabled is None else [bool(value) for value in args.partition_augmentation_train_enabled]
    if len(train_augmentation_enabled) != 5:
        raise ValueError("partition-augmentation-train-enabled must contain five values")
    sampling_class_enabled = [True] * 5 if args.sampling_class_enabled is None else [bool(value) for value in args.sampling_class_enabled]
    loss_class_enabled = [True] * 5 if args.class_loss_enabled is None else [bool(value) for value in args.class_loss_enabled]
    if len(sampling_class_enabled) != 5 or len(loss_class_enabled) != 5:
        raise ValueError("class weighting switches must contain five values")
    val_targets = validate_targets(args.partition_augmentation_targets_val, val_records)
    test_targets = validate_targets(args.partition_augmentation_targets_test, test_records) if test_records else None
    train_partition_manifest = val_partition_manifest = test_partition_manifest = None
    # Training targets are the single authoritative source of virtual training
    # records.  Validation/test augmentation is separately diagnostic only.
    train_records, train_partition_manifest = expand_trajectory_records(
        train_records, partition="train", targets=train_targets, seed=args.seed,
        frame_drop_fraction=args.augmentation_frame_drop_fraction,
        amplitude_scale_max_delta=args.augmentation_amplitude_scale_max_delta,
        snr_offset_db=args.augmentation_snr_offset_db,
        enabled=train_augmentation_enabled,
        method=args.partition_augmentation_method,
    )
    if args.partition_augmentation_diagnostics:
        val_augmented_records, val_partition_manifest = expand_trajectory_records(
            val_records, partition="val", targets=val_targets, seed=args.seed,
            frame_drop_fraction=args.augmentation_frame_drop_fraction,
            amplitude_scale_max_delta=args.augmentation_amplitude_scale_max_delta,
            snr_offset_db=args.augmentation_snr_offset_db,
            method=args.partition_augmentation_method,
        )
        if test_records:
            test_augmented_records, test_partition_manifest = expand_trajectory_records(
                test_records, partition="test", targets=test_targets, seed=args.seed,
                frame_drop_fraction=args.augmentation_frame_drop_fraction,
                amplitude_scale_max_delta=args.augmentation_amplitude_scale_max_delta,
                snr_offset_db=args.augmentation_snr_offset_db,
                method=args.partition_augmentation_method,
            )
        else:
            test_augmented_records = []
    else:
        val_augmented_records = val_records
        test_augmented_records = test_records
    write_json(output / "progress.json", {"phase": "preparing", "stage": "loading_tr_feature_cache", "epoch": 0, "total_epochs": args.epochs, "batch": 0, "total_batches": 0, "percent": 0.0, "train_trajectories": len(train_records), "val_trajectories": len(val_records)})
    cache_root = DEFAULT_TRAJECTORY_CACHE_ROOT
    train_dataset, train_cache = load_or_build(
        train_records, cache_root, {"partition_augmentation": train_partition_manifest["cache_parameters"]}
    )
    val_dataset, val_cache = load_or_build(
        val_records, cache_root, {"partition": "val", "method": "original"}
    )
    val_augmented_context = (
        {"partition_augmentation": val_partition_manifest["cache_parameters"]}
        if val_partition_manifest else {"partition": "val", "method": "original"}
    )
    val_augmented_dataset, val_augmented_cache = load_or_build(
        val_augmented_records, cache_root, val_augmented_context
    )
    write_json(output / "progress.json", {"phase": "preparing", "stage": "computing_source_stats", "epoch": 0, "total_epochs": args.epochs, "batch": 0, "total_batches": 0, "percent": 0.0, "train_trajectories": len(train_records), "val_trajectories": len(val_records)})
    center, scale = fit_source_stats(train_dataset)
    config = {
        "experiment_type": "tr_only_training",
        "experiment_label": "TR-only B01-compatible training",
        "epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "lr_scheduler": args.lr_scheduler,
        "patience": args.patience, "seed": args.seed, "dropout": args.dropout,
        "class_weight_mode": args.class_weight_mode, "class_balanced_beta": args.class_balanced_beta,
        "class_weight_floor": args.class_weight_floor, "class_weight_cap": args.class_weight_cap,
        "manual_class_loss_weights": list(args.manual_class_loss_weights) if args.manual_class_loss_weights is not None else None,
        "sampling_class_enabled": sampling_class_enabled,
        "class_loss_enabled": loss_class_enabled,
        "drone_sample_boost": args.drone_sample_boost,
        "bird_sample_boost": args.bird_sample_boost,
        "balloon_sample_boost": args.balloon_sample_boost,
        "clutter_sample_boost": args.clutter_sample_boost,
        "sampling_mode": args.sampling_mode, "sampling_protocol": args.sampling_protocol,
        "manual_sampling_boosts_enabled": bool(args.manual_sampling_boosts_enabled),
        "unknown_sample_boost": args.unknown_sample_boost,
        "checkpoint_selection_metric": args.checkpoint_selection_metric,
        "partition_augmentation_method": args.partition_augmentation_method,
        "trajectory_augmentation": {
            "train_only": True, "target_count_semantics": "per-class final train record count",
            "method": args.partition_augmentation_method,
            "targets": train_targets,
            "augmentation_enabled": train_augmentation_enabled,
            "frame_drop_fraction": args.augmentation_frame_drop_fraction,
            "amplitude_scale_max_delta": args.augmentation_amplitude_scale_max_delta,
            "snr_offset_db": args.augmentation_snr_offset_db,
            "base_counts": label_counts(base_train_records), "expanded_counts": label_counts(train_records),
            "derived_weights_recomputed_from_expanded_records": True,
            "source_normalization_computed_from": "expanded_train_records",
        }, "device": str(device),
        "partition_augmentation_diagnostics": {
            "enabled": bool(args.partition_augmentation_diagnostics),
            "method": args.partition_augmentation_method,
            "description": "validation/test related virtual trajectory diagnostic; not independent samples",
            "train": train_partition_manifest,
            "val": val_partition_manifest,
            "test": test_partition_manifest,
        },
        "source_normalization": {
            "enabled": True, "applies_to": "15-dimensional sequence input only",
            "rule": "(x - training_mean) / training_std", "computed_from": "expanded_train_records",
            "feature_count": int(center.numel()), "center": center.tolist(), "scale": scale.tolist(),
        },
        "use_cosface": bool(args.use_cosface),
        "sequence_encoder_type": "transformer", "cosface_scale": 16.0, "cosface_margin": 0.2,
        "split": {"manifest": str(args.split.resolve()), "sha256": file_sha256(args.split), "train": len(base_train_records), "train_expanded": len(train_records), "val": len(val_records), "test": len(test_records)},
        "track_index": str(args.track_index.resolve()), "partition": "val", "test_deferred": bool(args.skip_test),
        "resume": {"requested": str(args.resume.resolve()) if args.resume else None},
        "tr_feature_cache": {"train": train_cache, "val": val_cache, "val_augmented": val_augmented_cache},
    }
    write_json(output / "config.json", config)
    write_json(output / "partition_augmentation_manifest.json", config["partition_augmentation_diagnostics"])
    write_json(output / "progress.json", {"phase": "preparing", "stage": "building_dataloaders", "epoch": 0, "total_epochs": args.epochs, "batch": 0, "total_batches": 0, "percent": 0.0})
    sampling_info = sampling_weights(
        train_records, sampling_mode=args.sampling_mode, drone_boost=args.drone_sample_boost, bird_boost=args.bird_sample_boost,
        balloon_boost=args.balloon_sample_boost, clutter_boost=args.clutter_sample_boost,
        unknown_boost=args.unknown_sample_boost, class_enabled=sampling_class_enabled,
    )[1]
    sampling_info.update({"sampling_protocol": args.sampling_protocol,
                          "replacement": args.sampling_protocol == "strict_b01_replacement"})
    train_loader = make_loader(
        train_records, args.batch_size, args.workers, train=True, seed=args.seed, sampling_protocol=args.sampling_protocol, sampling_mode=args.sampling_mode,
        drone_boost=args.drone_sample_boost, bird_boost=args.bird_sample_boost,
        balloon_boost=args.balloon_sample_boost, clutter_boost=args.clutter_sample_boost,
        unknown_boost=args.unknown_sample_boost, class_enabled=sampling_class_enabled, dataset=train_dataset,
    )
    val_loader = make_loader(val_records, args.batch_size, args.workers, train=False, seed=args.seed, dataset=val_dataset)
    val_augmented_loader = make_loader(val_augmented_records, args.batch_size, args.workers, train=False, seed=args.seed, dataset=val_augmented_dataset)
    model = TrajectoryBranch(dropout=args.dropout).to(device)
    model.source_center.copy_(center)
    model.source_scale.copy_(scale)
    counts = torch.bincount(torch.tensor([record.label for record in train_records]), minlength=5)
    class_weight_values = select_class_weights(counts, args)
    config["sampling"] = sampling_info
    config["class_loss"] = {
        "rule": args.class_weight_mode, "computed_from": "expanded_train_records",
        "class_counts": {str(index): int(value) for index, value in enumerate(counts.tolist())},
        "weights": {str(index): float(value) for index, value in enumerate(class_weight_values.tolist())},
        "weight_source": "manual_override" if args.manual_class_loss_weights is not None else "automatic_rule",
        "manual_override": {str(index): float(value) for index, value in enumerate(args.manual_class_loss_weights)}
        if args.manual_class_loss_weights is not None else None,
    }
    write_json(output / "config.json", config)
    class_weights = class_weight_values.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    encoder_ids = {id(parameter) for parameter in model.representation_builder.sequence_encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in encoder_ids], "lr": args.learning_rate * 0.1, "weight_decay": args.weight_decay},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": args.learning_rate, "weight_decay": args.weight_decay},
    ])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
                 if args.lr_scheduler == "cosine" else None)
    history: list[dict[str, Any]] = []
    start_epoch, best_epoch, best_f1, best_selection_score, stale = 1, 0, float("-inf"), float("-inf"), 0
    if args.resume:
        resume_path = args.resume.resolve()
        source_dir = resume_path.parent
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state") or checkpoint.get("model_state_dict"), strict=True)
        if checkpoint.get("source_center") is not None: model.source_center.copy_(checkpoint["source_center"])
        if checkpoint.get("source_scale") is not None: model.source_scale.copy_(checkpoint["source_scale"])
        source_history = output / "history.json"
        if not source_history.is_file():
            source_history = source_dir / "history.json"
        history = checkpoint.get("history") or (json.loads(source_history.read_text(encoding="utf-8")) if source_history.is_file() else [])
        completed_epoch = int(checkpoint.get("epoch", 0))
        best_epoch = int(checkpoint.get("best_epoch", 0)); best_f1 = float(checkpoint.get("best_val_macro_f1", float("-inf"))); best_selection_score = float(checkpoint.get("best_selection_score", best_f1)); stale = int(checkpoint.get("stale_epochs", 0))
        if checkpoint.get("optimizer_state") and (scheduler is None or checkpoint.get("scheduler_state")):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None: scheduler.load_state_dict(checkpoint["scheduler_state"])
            restore_rng(checkpoint.get("rng_state") or {})
        elif scheduler is not None:
            scheduler.last_epoch = completed_epoch
        start_epoch = completed_epoch + 1
        config["resume"].update({"mode": "exact_last_epoch" if checkpoint.get("optimizer_state") else "legacy_best_warm_start", "completed_epoch": completed_epoch, "next_epoch": start_epoch, "best_epoch": best_epoch})
        write_json(output / "config.json", config)
        source_best = source_dir / "best.pt"
        source_validation = source_dir / "validation_best.json"
        if source_best.is_file() and source_best.resolve() != (output / "best.pt").resolve():
            shutil.copy2(source_best, output / "best.pt")
        if source_validation.is_file():
            shutil.copy2(source_validation, output / "validation_best.json")
    best_records: list[dict[str, Any]] = []
    best_augmented_records: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); loss_sum = 0.0; correct = 0; count = 0
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        total_batches = len(train_loader)
        write_json(output / "progress.json", {"phase": "train", "epoch": epoch, "total_epochs": args.epochs, "batch": 0, "total_batches": total_batches, "percent": (epoch - 1) / args.epochs * 100.0})
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            sequence = batch.sequence.to(device, non_blocking=True); physical = batch.physical.to(device, non_blocking=True); mask = batch.padding_mask.to(device, non_blocking=True); labels = batch.labels.to(device, non_blocking=True)
            logits = model(sequence, physical, mask, labels=labels if args.use_cosface else None); train_loss = criterion(logits, labels); train_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            batch_count = int(labels.numel())
            batch_correct = int((logits.argmax(dim=-1) == labels).sum().item())
            loss_sum += float(train_loss.detach().item()) * batch_count; correct += batch_correct; count += batch_count
            train_acc = correct / max(count, 1)
            write_json(output / "progress.json", {"phase": "train", "epoch": epoch, "total_epochs": args.epochs, "batch": batch_index, "total_batches": total_batches, "percent": ((epoch - 1) + batch_index / max(total_batches, 1)) / args.epochs * 100.0, "train_loss": loss_sum / max(count, 1), "train_frame_accuracy": train_acc, "learning_rate": optimizer.param_groups[1]["lr"]})
        validation, records = evaluate(model, val_loader, device, criterion, output, epoch, args.epochs)
        validation_augmented = None
        validation_augmented_records: list[dict[str, Any]] = []
        if args.partition_augmentation_diagnostics:
            validation_augmented, validation_augmented_records = evaluate(model, val_augmented_loader, device, criterion, output, epoch, args.epochs, "validation_augmented")
        train_loss_value = loss_sum / max(count, 1)
        train_acc = correct / max(count, 1)
        sampler_audit = getattr(train_loader.sampler, "last_audit", {})
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[1]["lr"],
            "train_loss": train_loss_value,
            "train_frame_accuracy": train_acc,
            "train_uses_cosface": bool(args.use_cosface),
            "val_loss": validation["loss"],
            "val_trajectory_macro_f1": validation["macro_f1"],
            "val_trajectory_accuracy": validation["accuracy"],
            "sampling_audit": sampler_audit,
        }
        if validation_augmented is not None:
            row.update({"val_augmented_loss": validation_augmented["loss"], "val_augmented_trajectory_macro_f1": validation_augmented["macro_f1"], "val_augmented_trajectory_accuracy": validation_augmented["accuracy"]})
        selection_score = float(validation["macro_f1"])
        if args.checkpoint_selection_metric == "bird_f1":
            report = validation.get("classification_report", {})
            bird_f1 = float(report.get("bird", {}).get("f1-score", 0.0))
            drone_f1 = float(report.get("drone", {}).get("f1-score", 0.0))
            selection_score = bird_f1 + 1e-6 * float(validation["macro_f1"]) + 1e-9 * drone_f1
        row.update({"checkpoint_selection_metric": args.checkpoint_selection_metric, "selection_score": selection_score})
        history.append(row); write_json(output / "history.json", history); write_json(output / "validation_latest.json", validation)
        if validation_augmented is not None: write_json(output / "validation_augmented_latest.json", validation_augmented)
        write_json(output / "progress.json", {"phase": "validation", "epoch": epoch, "total_epochs": args.epochs, "batch": len(val_loader), "total_batches": len(val_loader), "percent": epoch / args.epochs * 100.0, **row})
        if selection_score > best_selection_score:
            best_selection_score = selection_score; best_f1 = float(validation["macro_f1"]); best_epoch = epoch; stale = 0; best_records = records; best_augmented_records = validation_augmented_records
            checkpoint_for_model(output / "best.pt", model, epoch, best_f1, best_selection_score, center, scale, config)
            write_json(output / "validation_best.json", validation)
            if validation_augmented is not None: write_json(output / "validation_augmented_best.json", validation_augmented)
        else:
            stale += 1
        if scheduler is not None: scheduler.step()
        save_last(output / "last.pt", model, optimizer, scheduler, epoch, best_epoch, best_f1, best_selection_score, stale, history, center, scale, config)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.patience > 0 and stale >= args.patience: break
    if not (output / "best.pt").is_file():
        raise RuntimeError("TR training ended without a best checkpoint")
    best_checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint.get("model_state") or best_checkpoint.get("model_state_dict"), strict=True)
    if not best_records:
        validation, best_records = evaluate(model, val_loader, device, criterion, output, best_epoch or args.epochs, args.epochs)
    if args.partition_augmentation_diagnostics and not best_augmented_records:
        _, best_augmented_records = evaluate(model, val_augmented_loader, device, criterion, output, best_epoch or args.epochs, args.epochs, "validation_augmented")
    write_json(output / "validation_best.json", (read_json(output / "validation_best.json") or {}))
    with (output / "trajectory_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for record in best_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.partition_augmentation_diagnostics:
        with (output / "trajectory_decisions_augmented.jsonl").open("w", encoding="utf-8") as handle:
            for record in best_augmented_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.skip_test:
        write_json(output / "ablation_complete.json", {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "test_deferred": True})
    else:
        test_dataset, test_cache = load_or_build(
            test_records, cache_root, {"partition": "test", "method": "original"}
        )
        config["tr_feature_cache"]["test"] = test_cache; write_json(output / "config.json", config)
        test_loader = make_loader(test_records, args.batch_size, args.workers, train=False, seed=args.seed, dataset=test_dataset)
        test_metrics, test_records_out = evaluate(model, test_loader, device, criterion, output, best_epoch or args.epochs, args.epochs, "testing")
        write_json(output / "test_trajectory_metrics.json", test_metrics)
        with (output / "trajectory_decisions_test.jsonl").open("w", encoding="utf-8") as handle:
            for record in test_records_out: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.partition_augmentation_diagnostics:
            test_augmented_dataset, test_augmented_cache = load_or_build(
                test_augmented_records, cache_root,
                {"partition_augmentation": test_partition_manifest["cache_parameters"]},
            )
            config["tr_feature_cache"]["test_augmented"] = test_augmented_cache; write_json(output / "config.json", config)
            test_augmented_loader = make_loader(test_augmented_records, args.batch_size, args.workers, train=False, seed=args.seed, dataset=test_augmented_dataset)
            test_augmented_metrics, test_augmented_records_out = evaluate(model, test_augmented_loader, device, criterion, output, best_epoch or args.epochs, args.epochs, "testing_augmented")
            write_json(output / "test_augmented_trajectory_metrics.json", test_augmented_metrics)
            with (output / "trajectory_decisions_test_augmented.jsonl").open("w", encoding="utf-8") as handle:
                for record in test_augmented_records_out: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(output / "ablation_complete.json", {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "test_deferred": False})
    write_json(output / "progress.json", {"phase": "complete", "epoch": best_epoch, "total_epochs": args.epochs, "batch": 0, "total_batches": 0, "percent": 100.0, "best_epoch": best_epoch, "best_val_macro_f1": best_f1})


def read_json(path: Path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return None


if __name__ == "__main__":
    main()

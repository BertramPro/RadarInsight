"""Evaluate B01 TR, RD, and their trajectory-level soft cascade on one fixed split."""

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
from radar_fusion.model import (  # noqa: E402
    CLASS_NAMES,
    SoftCascadeFusion,
    load_checkpoint_metadata,
    load_b01_trajectory_branch,
)
from radar_fusion.reporting import save_fusion_report  # noqa: E402
from radar_fusion.ml_track import feature_matrix, load_bundle  # noqa: E402
from radar_fusion.partition_augmentation import expand_trajectory_records, expand_rd_frames, validate_targets  # noqa: E402
from radar_fusion.trajectory_cache import DEFAULT_TRAJECTORY_CACHE_ROOT, load_or_build  # noqa: E402
from radar_rd.train import RDCache, RDDataset, SmallRDCNN, build_manifest, load_rd  # noqa: E402


DEFAULT_SPLIT = Path(r"K:\radar\main\data\manifests\cq08_grouped_split_f.json")
DEFAULT_TRACK_INDEX = Path(r"K:\radar\main\data\processed\expert1_track_index.csv")
DEFAULT_TR_CHECKPOINT = Path(
    r"K:\radar\main\artifacts\f_protocol\20260728-183147\b01_transformer_seed42\best_model.pt"
)
DEFAULT_RD_CHECKPOINT = PROJECT_ROOT / "artifacts" / "rd_ablation_R2_contrast_w900_registry_seed42_rerun" / "best.pt"


class HybridRDCache:
    """Use compatible cached frames and preprocess only cache misses."""

    def __init__(self, cache_dir: Path | None, config: dict[str, object]) -> None:
        self.cache_dir = cache_dir.expanduser().resolve() if cache_dir is not None else None
        self.index: dict[str, int] = {}
        self.images = None
        self.observed = None
        self.cache: RDCache | None = None
        self.hits = 0
        self.misses = 0
        self.config = config
        if self.cache_dir is None:
            return
        # A fusion evaluation may legitimately ask for a held-out partition
        # absent from a train/validation cache.  RDCache validates the cache
        # artifact itself, while allow_missing keeps those frames on the exact
        # same deterministic preprocessing path instead of treating a miss as
        # a different input representation.
        self.cache = RDCache(
            self.cache_dir, (),
            velocity_min=float(config["velocity_min"]),
            velocity_max=float(config["velocity_max"]),
            target_width=int(config["target_width"]),
            resampling=str(config["resampling"]),
            allow_missing=True,
        )
        self.index = self.cache.index
        self.images = self.cache.images
        self.observed = self.cache.observed

    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        position = self.index.get(path)
        if position is not None:
            self.hits += 1
            assert self.cache is not None
            return self.cache.load(path)
        self.misses += 1
        if self.cache is not None:
            return self.cache.load(path)
        return load_rd(
            path,
            float(self.config["velocity_min"]),
            float(self.config["velocity_max"]),
            int(self.config["target_width"]),
            str(self.config["resampling"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--track-index", type=Path, default=DEFAULT_TRACK_INDEX)
    parser.add_argument("--tr-checkpoint", type=Path, default=DEFAULT_TR_CHECKPOINT)
    parser.add_argument("--rd-checkpoint", type=Path, default=DEFAULT_RD_CHECKPOINT)
    parser.add_argument("--rd-cache", type=Path, default=None)
    parser.add_argument("--partition", choices=["train", "val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size-tr", type=int, default=32)
    parser.add_argument("--batch-size-rd", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fusion-mode", choices=["fixed", "quality_classwise"], default="fixed")
    parser.add_argument("--fixed-rd-weight", type=float, nargs="+", default=[0.2])
    parser.add_argument("--fusion-checkpoint", type=Path, default=None)
    parser.add_argument("--partition-augmentation-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--partition-augmentation-method", choices=("perturbation", "smote"), default="perturbation")
    parser.add_argument("--partition-augmentation-targets-train", type=int, nargs=5, default=None)
    parser.add_argument("--partition-augmentation-targets-val", type=int, nargs=5, default=None)
    parser.add_argument("--partition-augmentation-targets-test", type=int, nargs=5, default=None)
    parser.add_argument("--augment-existing", action="store_true",
                        help="Keep the existing original fusion report and only append the virtual-trajectory diagnostic")
    return parser.parse_args()


def write_progress(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fusion_progress_plan(include_augmentation: bool, supplement_only: bool = False) -> list[str]:
    """Return the ordered, user-visible stages for a fusion evaluation."""
    if supplement_only:
        return ["fusion_augmented_tr", "fusion_augmented_rd", "fusion_augmented_combine"]
    stages = ["fusion_original_tr", "fusion_original_rd", "fusion_original_combine"]
    if include_augmentation:
        stages.extend(("fusion_augmented_tr", "fusion_augmented_rd", "fusion_augmented_combine"))
    return stages


def write_fusion_progress(
    args: argparse.Namespace,
    stages: list[str],
    phase: str,
    *,
    batch: int | None = None,
    total_batches: int | None = None,
    stage_percent: float | None = None,
) -> None:
    """Persist per-stage and whole-evaluation progress for the monitor."""
    if phase not in stages:
        raise ValueError(f"Unknown fusion progress phase: {phase}")
    stage_index = stages.index(phase) + 1
    if total_batches and total_batches > 0:
        percent = 100.0 * float(batch or 0) / float(total_batches)
    else:
        percent = float(stage_percent or 0.0)
    percent = max(0.0, min(100.0, percent))
    payload: dict[str, object] = {
        "phase": phase,
        "stage_index": stage_index,
        "stage_total": len(stages),
        "stage_percent": percent,
        "overall_percent": 100.0 * ((stage_index - 1) + percent / 100.0) / len(stages),
    }
    if total_batches is not None:
        payload["batch"] = int(batch or 0)
        payload["total_batches"] = int(total_batches)
    write_progress(args.output_dir, payload)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate_provenance(args: argparse.Namespace) -> dict[str, object] | None:
    """Reject a quality gate trained from a different split or branch weights."""
    if args.fusion_mode != "quality_classwise":
        return None
    assert args.fusion_checkpoint is not None
    payload = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if not isinstance(provenance, dict):
        raise ValueError("quality gate checkpoint has no provenance metadata; retrain the gate with matched TR/RD checkpoints")
    expected = {
        "split_sha256": file_sha256(args.split),
        "tr_checkpoint_sha256": file_sha256(args.tr_checkpoint),
        "rd_checkpoint_sha256": file_sha256(args.rd_checkpoint),
    }
    aliases = {"split_sha256": ("split_sha256", "grouped_split_sha256")}
    for key, value in expected.items():
        candidates = aliases.get(key, (key,))
        actual = next((provenance.get(name) for name in candidates if provenance.get(name)), None)
        if actual != value:
            raise ValueError(f"quality gate provenance mismatch for {key}: expected {value}, got {actual}")
    return provenance


@torch.inference_mode()
def infer_tr(
    args: argparse.Namespace,
    split: dict[str, str],
    device: torch.device,
    records=None,
    progress_stages: list[str] | None = None,
    progress_phase: str | None = None,
    cache_context: dict[str, object] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    records = records if records is not None else load_trajectory_records(args.track_index, split, args.partition)
    if args.tr_checkpoint.suffix.lower() in {".joblib", ".pkl"}:
        bundle = load_bundle(args.tr_checkpoint)
        matrix, truth, _, cache_metadata = feature_matrix(records, partition=args.partition)
        median = np.asarray(bundle.get("median", np.zeros(matrix.shape[1])), dtype=np.float64)
        matrix = np.where(np.isfinite(matrix), matrix, median)
        probabilities = np.asarray(bundle["model"].predict_proba(matrix), dtype=np.float64)
        logits_by_id = {record.trajectory_id: np.log(np.clip(probability, 1e-8, 1.0))
                        for record, probability in zip(records, probabilities)}
        labels = {record.trajectory_id: int(label) for record, label in zip(records, truth)}
        if not hasattr(args, "tr_feature_cache"): args.tr_feature_cache = {}
        args.tr_feature_cache[progress_phase or "trajectory_inference"] = cache_metadata
        if progress_stages is not None and progress_phase is not None:
            write_fusion_progress(args, progress_stages, progress_phase, batch=1, total_batches=1)
        else:
            write_progress(args.output_dir, {"phase": "trajectory_inference", "batch": 1, "total_batches": 1})
        return logits_by_id, labels
    if progress_stages is not None and progress_phase is not None:
        write_fusion_progress(args, progress_stages, progress_phase, batch=0, total_batches=1)
    if cache_context is None:
        cache_context = {"partition": args.partition, "method": "original"}
    dataset, cache_metadata = load_or_build(records, DEFAULT_TRAJECTORY_CACHE_ROOT, cache_context)
    cache_key = progress_phase or "trajectory_inference"
    if not hasattr(args, "tr_feature_cache"):
        args.tr_feature_cache = {}
    args.tr_feature_cache[cache_key] = cache_metadata
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_tr,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_trajectories,
        pin_memory=device.type == "cuda",
    )
    if progress_stages is not None and progress_phase is not None:
        write_fusion_progress(args, progress_stages, progress_phase, batch=0, total_batches=len(loader))
    model = load_b01_trajectory_branch(args.tr_checkpoint).to(device).eval()
    logits_by_id: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for batch_index, batch in enumerate(loader, start=1):
        logits = model(
            batch.sequence.to(device, non_blocking=True),
            batch.physical.to(device, non_blocking=True),
            batch.padding_mask.to(device, non_blocking=True),
        ).cpu().numpy()
        for trajectory_id, label, values in zip(batch.trajectory_ids, batch.labels.tolist(), logits):
            logits_by_id[trajectory_id] = values
            labels[trajectory_id] = int(label)
        if progress_stages is not None and progress_phase is not None:
            write_fusion_progress(args, progress_stages, progress_phase, batch=batch_index, total_batches=len(loader))
        else:
            write_progress(
                args.output_dir,
                {"phase": "trajectory_inference", "batch": batch_index, "total_batches": len(loader)},
            )
    return logits_by_id, labels


@torch.inference_mode()
def infer_rd(
    args: argparse.Namespace,
    split: dict[str, str],
    device: torch.device,
    frames=None,
    progress_stages: list[str] | None = None,
    progress_phase: str | None = None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, int], HybridRDCache, dict[str, object]]:
    checkpoint = load_checkpoint_metadata(args.rd_checkpoint)
    config_path = args.rd_checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frames = frames if frames is not None else [frame for frame in build_manifest(args.dataset_root) if split.get(frame.trajectory_id) == args.partition]
    cache_path = args.rd_cache or (Path(config["rd_cache"]) if config.get("rd_cache") else None)
    cache = HybridRDCache(cache_path, config)
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
        derived_cache_dir=PROJECT_ROOT / "cache" / "rd_partition_augmented",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_rd,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    if progress_stages is not None and progress_phase is not None:
        write_fusion_progress(args, progress_stages, progress_phase, batch=0, total_batches=len(loader))
    input_channels = 1 if config["input_mode"] == "rd" else 2
    model = SmallRDCNN(input_channels=input_channels, head=str(config["model_head"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    logits_by_id: dict[str, list[np.ndarray]] = {}
    labels: dict[str, int] = {}
    for batch_index, (images, targets, trajectory_ids) in enumerate(loader, start=1):
        logits = model(images.to(device, non_blocking=True)).cpu().numpy()
        for trajectory_id, label, values in zip(trajectory_ids, targets.tolist(), logits):
            logits_by_id.setdefault(str(trajectory_id), []).append(values)
            labels[str(trajectory_id)] = int(label)
        if progress_stages is not None and progress_phase is not None:
            write_fusion_progress(args, progress_stages, progress_phase, batch=batch_index, total_batches=len(loader))
        else:
            write_progress(
                args.output_dir,
                {"phase": "rd_inference", "batch": batch_index, "total_batches": len(loader)},
            )
    return logits_by_id, labels, cache, config


def fused_records(args, device, expected_ids, tr_logits_by_id, labels, rd_logits_by_id):
    fixed_weight = args.fixed_rd_weight[0] if len(args.fixed_rd_weight) == 1 else args.fixed_rd_weight
    tr_logits = torch.tensor(np.stack([tr_logits_by_id[trajectory_id] for trajectory_id in expected_ids]), dtype=torch.float32)
    frame_logits, frame_to_track = [], []
    for track_index, trajectory_id in enumerate(expected_ids):
        values = rd_logits_by_id[trajectory_id]; frame_logits.extend(values); frame_to_track.extend([track_index] * len(values))
    fusion = SoftCascadeFusion(mode=args.fusion_mode, fixed_rd_weight=fixed_weight)
    if args.fusion_checkpoint is not None:
        payload = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
        fusion.load_state_dict(payload.get("fusion_state", payload), strict=True)
    output = fusion(tr_logits, torch.tensor(np.stack(frame_logits), dtype=torch.float32), torch.tensor(frame_to_track, dtype=torch.long))
    records = []
    for index, trajectory_id in enumerate(expected_ids):
        truth = labels[trajectory_id]; tr_prediction = int(output.tr_predictions[index]); rd_prediction = int(output.rd_predictions[index]); fused_prediction = int(output.fused_predictions[index])
        records.append({"trajectory_id": trajectory_id, "true_class": truth, "true_label": CLASS_NAMES[truth],
                        "tr_logits": output.tr_logits[index].tolist(), "tr_probabilities": output.tr_probabilities[index].tolist(), "tr_prediction": tr_prediction, "tr_prediction_label": CLASS_NAMES[tr_prediction],
                        "rd_probabilities": output.rd_probabilities[index].tolist(), "rd_prediction": rd_prediction, "rd_prediction_label": CLASS_NAMES[rd_prediction],
                        "rd_frame_count": int(output.rd_frame_count[index]), "rd_consistency": float(output.rd_consistency[index]), "rd_available": bool(output.rd_available[index]), "rd_class_weights": output.rd_class_weights[index].tolist(),
                        "fused_probabilities": output.fused_probabilities[index].tolist(), "fused_prediction": fused_prediction, "fused_prediction_label": CLASS_NAMES[fused_prediction],
                        "branch_agreement": tr_prediction == rd_prediction, "fusion_rescue_vs_tr": tr_prediction != truth and fused_prediction == truth, "fusion_harm_vs_tr": tr_prediction == truth and fused_prediction != truth})
    return records


def main() -> None:
    args = parse_args()
    if args.partition == "test" and not args.allow_test:
        raise ValueError("test evaluation is deferred; pass --allow-test only after model selection is complete")
    if args.fusion_mode == "quality_classwise" and args.fusion_checkpoint is None:
        raise ValueError("quality_classwise fusion requires a trained --fusion-checkpoint")
    gate_provenance = validate_gate_provenance(args)
    fixed_weight: float | list[float]
    fixed_weight = args.fixed_rd_weight[0] if len(args.fixed_rd_weight) == 1 else args.fixed_rd_weight
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    monitor_config = {
        "experiment_type": "tr_rd_soft_cascade",
        "experiment_label": "TR-RD trajectory-level soft cascade",
        "dataset_root": str(args.dataset_root.resolve()),
        "grouped_split": str(args.split.resolve()),
        "track_index": str(args.track_index.resolve()),
        "tr_checkpoint": str(args.tr_checkpoint.resolve()),
        "rd_checkpoint": str(args.rd_checkpoint.resolve()),
        "rd_cache": str(args.rd_cache.resolve()) if args.rd_cache else "",
        "partition": args.partition,
        "batch_size_tr": args.batch_size_tr,
        "batch_size_rd": args.batch_size_rd,
        "workers": args.workers,
        "device": str(device),
        "fusion_mode": args.fusion_mode,
        "fixed_rd_weight": args.fixed_rd_weight,
        "fusion_checkpoint": str(args.fusion_checkpoint.resolve()) if args.fusion_checkpoint else "",
        "gate_provenance": gate_provenance,
        "test_deferred": args.partition != "test",
        "partition_augmentation_method": args.partition_augmentation_method,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(monitor_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    progress_stages = fusion_progress_plan(
        args.partition_augmentation_diagnostics,
        supplement_only=args.augment_existing,
    )
    write_progress(args.output_dir, {
        "phase": "fusion_preparing",
        "stage_index": 0,
        "stage_total": len(progress_stages),
        "stage_percent": 0.0,
        "overall_percent": 0.0,
    })
    split = load_grouped_split(args.split)
    tr_base_records = load_trajectory_records(args.track_index, split, args.partition)
    rd_base_frames = [frame for frame in build_manifest(args.dataset_root) if split.get(frame.trajectory_id) == args.partition]
    requested = {"train": args.partition_augmentation_targets_train, "val": args.partition_augmentation_targets_val, "test": args.partition_augmentation_targets_test}[args.partition]
    targets = validate_targets(requested, tr_base_records)
    tr_augmented_records, tr_manifest = expand_trajectory_records(
        tr_base_records, partition=args.partition, targets=targets, seed=42, allow_frame_drop=False,
        method=args.partition_augmentation_method,
    ) if args.partition_augmentation_diagnostics else (tr_base_records, None)
    rd_augmented_frames, rd_manifest = expand_rd_frames(
        rd_base_frames, partition=args.partition, targets=targets, seed=42,
        method=args.partition_augmentation_method,
        smote_plan=tr_manifest if args.partition_augmentation_method == "smote" else None,
    ) if args.partition_augmentation_diagnostics else (rd_base_frames, None)
    if args.augment_existing:
        if not args.partition_augmentation_diagnostics:
            raise ValueError("--augment-existing requires partition augmentation diagnostics")
        existing_config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8")) if (args.output_dir / "config.json").is_file() else None
        if not isinstance(existing_config, dict):
            raise ValueError("Existing fusion evaluation is missing config.json")
        augmented_metrics_path = args.output_dir / "augmented_metrics.json"
        augmented_decisions_path = args.output_dir / "trajectory_decisions_augmented.jsonl"
        if augmented_metrics_path.exists() or augmented_decisions_path.exists():
            raise ValueError("This fusion evaluation already contains the augmentation diagnostic")
        tr_augmented_logits, tr_augmented_labels = infer_tr(
            args, split, device, tr_augmented_records, progress_stages, "fusion_augmented_tr",
            {"partition_augmentation": tr_manifest["cache_parameters"]},
        )
        rd_augmented_logits, rd_augmented_labels, rd_cache, rd_config = infer_rd(
            args, split, device, rd_augmented_frames, progress_stages, "fusion_augmented_rd"
        )
        augmented_ids = sorted(tr_augmented_labels, key=str)
        if set(augmented_ids) != set(rd_augmented_labels):
            raise ValueError("augmented TR/RD virtual ids do not match")
        write_fusion_progress(args, progress_stages, "fusion_augmented_combine", stage_percent=0.0)
        augmented_records = fused_records(args, device, augmented_ids, tr_augmented_logits, tr_augmented_labels, rd_augmented_logits)
        existing_metrics = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8")) if (args.output_dir / "metrics.json").is_file() else {}
        provenance = ((existing_metrics.get("provenance") or existing_config.get("provenance") or {})
                      if isinstance(existing_config, dict) else {})
        augmented_summary = save_fusion_report(args.output_dir / "augmented_diagnostic", augmented_records,
                                               provenance | {"partition_augmentation": True})
        augmented_metrics_path.write_text(json.dumps(augmented_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        augmented_decisions_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in augmented_records), encoding="utf-8")
        manifest = {"enabled": True, "method": args.partition_augmentation_method,
                    "description": "related virtual trajectory diagnostic; not independent samples",
                    "tr": tr_manifest, "rd": rd_manifest}
        (args.output_dir / "partition_augmentation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        supplement = {"kind": "partition_augmentation_diagnostic", "partition": args.partition,
                      "completed_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                      "trajectory_count": augmented_summary.get("soft_cascade", {}).get("trajectory_count", 0)}
        (args.output_dir / "augmentation_supplement.json").write_text(json.dumps(supplement, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        merged_config = dict(existing_config)
        merged_config["partition_augmentation_diagnostics"] = manifest
        merged_config["augmentation_supplement"] = supplement
        merged_config["tr_feature_cache"] = dict(getattr(args, "tr_feature_cache", {}))
        (args.output_dir / "config.json").write_text(json.dumps(merged_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_fusion_progress(args, progress_stages, "fusion_augmented_combine", stage_percent=100.0)
        write_progress(args.output_dir, {"phase": "complete", "percent": 100.0, "overall_percent": 100.0,
                                         "augmentation_supplement": True})
        print(json.dumps({"augmentation": True, "fused_macro_f1": augmented_summary["soft_cascade"]["macro_f1"],
                          "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))
        return
    monitor_config["partition_augmentation_diagnostics"] = {
        "enabled": bool(args.partition_augmentation_diagnostics),
        "method": args.partition_augmentation_method,
        "description": "related virtual trajectory diagnostic; not independent samples",
        "tr": tr_manifest, "rd": rd_manifest,
    }
    (args.output_dir / "config.json").write_text(json.dumps(monitor_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tr_logits_by_id, tr_labels = infer_tr(args, split, device, progress_stages=progress_stages,
                                           progress_phase="fusion_original_tr")
    rd_logits_by_id, rd_labels, rd_cache, rd_config = infer_rd(args, split, device,
                                                                progress_stages=progress_stages,
                                                                progress_phase="fusion_original_rd")
    expected_ids = sorted(
        (trajectory_id for trajectory_id, value in split.items() if value == args.partition),
        key=int,
    )
    if set(tr_logits_by_id) != set(expected_ids) or set(rd_logits_by_id) != set(expected_ids):
        raise ValueError("TR/RD evidence does not exactly cover the requested grouped split")
    if tr_labels != rd_labels:
        mismatches = [trajectory_id for trajectory_id in expected_ids if tr_labels[trajectory_id] != rd_labels[trajectory_id]]
        raise ValueError(f"TR/RD labels disagree for {len(mismatches)} trajectories (e.g. {mismatches[0]})")

    write_fusion_progress(args, progress_stages, "fusion_original_combine", stage_percent=0.0)
    records = fused_records(args, device, expected_ids, tr_logits_by_id, tr_labels, rd_logits_by_id)
    write_fusion_progress(args, progress_stages, "fusion_original_combine", stage_percent=100.0)
    augmented_records = []
    if args.partition_augmentation_diagnostics:
        tr_augmented_logits, tr_augmented_labels = infer_tr(
            args, split, device, tr_augmented_records, progress_stages, "fusion_augmented_tr",
            {"partition_augmentation": tr_manifest["cache_parameters"]},
        )
        rd_augmented_logits, rd_augmented_labels, _, _ = infer_rd(
            args, split, device, rd_augmented_frames, progress_stages, "fusion_augmented_rd"
        )
        augmented_ids = sorted(tr_augmented_labels, key=str)
        if set(augmented_ids) != set(rd_augmented_labels): raise ValueError("augmented TR/RD virtual ids do not match")
        write_fusion_progress(args, progress_stages, "fusion_augmented_combine", stage_percent=0.0)
        augmented_records = fused_records(args, device, augmented_ids, tr_augmented_logits, tr_augmented_labels, rd_augmented_logits)
        write_fusion_progress(args, progress_stages, "fusion_augmented_combine", stage_percent=100.0)

    rd_source_split_path = args.rd_checkpoint.parent / "split.json"
    rd_source_split = load_grouped_split(rd_source_split_path) if rd_source_split_path.is_file() else {}
    provenance = {
        "partition": args.partition,
        "validation_only": args.partition == "val",
        "grouped_split": str(args.split.resolve()),
        "grouped_split_sha256": file_sha256(args.split),
        "tr_checkpoint": str(args.tr_checkpoint.resolve()),
        "tr_checkpoint_epoch": (0 if args.tr_checkpoint.suffix.lower() in {".joblib", ".pkl"}
                                else int(load_checkpoint_metadata(args.tr_checkpoint).get("epoch", 0))),
        "tr_implementation": "ml" if args.tr_checkpoint.suffix.lower() in {".joblib", ".pkl"} else "dl",
        "rd_checkpoint": str(args.rd_checkpoint.resolve()),
        "rd_checkpoint_epoch": int(load_checkpoint_metadata(args.rd_checkpoint).get("epoch", 0)),
        "rd_training_membership_matches_f_split": all(
            (value != "train") or rd_source_split.get(trajectory_id) == "train"
            for trajectory_id, value in split.items()
        ),
        "rd_saved_validation_membership_matches_f_split": rd_source_split == split,
        "fusion_mode": args.fusion_mode,
        "fixed_rd_weight": args.fixed_rd_weight if args.fusion_mode == "fixed" else None,
        "fusion_checkpoint": str(args.fusion_checkpoint.resolve()) if args.fusion_checkpoint else None,
        "rd_configuration": {
            key: rd_config[key]
            for key in ("velocity_min", "velocity_max", "target_width", "resampling", "normalization", "input_mode", "model_head")
        },
        "rd_cache": str(rd_cache.cache_dir) if rd_cache.cache_dir else None,
        "rd_cache_hits": rd_cache.hits,
        "rd_cache_misses": rd_cache.misses,
        "tr_feature_cache": dict(getattr(args, "tr_feature_cache", {})),
    }
    monitor_config["tr_feature_cache"] = dict(getattr(args, "tr_feature_cache", {}))
    (args.output_dir / "config.json").write_text(
        json.dumps(monitor_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = save_fusion_report(args.output_dir, records, provenance)
    if augmented_records:
        augmented_summary = save_fusion_report(args.output_dir / "augmented_diagnostic", augmented_records, provenance | {"partition_augmentation": True})
        (args.output_dir / "augmented_metrics.json").write_text(json.dumps(augmented_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "trajectory_decisions_augmented.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in augmented_records), encoding="utf-8")
        (args.output_dir / "partition_augmentation_manifest.json").write_text(json.dumps(monitor_config["partition_augmentation_diagnostics"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.partition == "test":
        # Keep final held-out results distinct from validation reports while
        # retaining metrics.json as the monitor's fusion report contract.
        (args.output_dir / "test_trajectory_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    write_progress(args.output_dir, {"phase": "complete", "percent": 100.0, "overall_percent": 100.0})
    print(json.dumps({
        "tr_macro_f1": summary["tr_branch"]["macro_f1"],
        "rd_macro_f1": summary["rd_branch"]["macro_f1"],
        "fused_macro_f1": summary["soft_cascade"]["macro_f1"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

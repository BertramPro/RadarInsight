import json
import hashlib
import os
import re
import subprocess
import sys
import threading
import time
import copy
from urllib.parse import parse_qs, unquote, urlparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Union

from training_scheduler import ACTIVE_STATES, CONFIRMATION_STATE, TERMINAL_STATES, TrainingScheduler

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "scripts" / "training_monitor.html"
EXPERIMENT_INDEX_PATH = ROOT / "tmp" / "training_experiment_index.json"
EXPERIMENT_INDEX_LOCK = threading.Lock()
ETA_SAMPLES = {}
ETA_LOCK = threading.Lock()
RESOURCE_LOCK = threading.Lock()
RESOURCE_CACHE = {"time": 0.0, "value": None}
PROCESS_CPU_CACHE = {}
SYSTEM_CPU_CACHE = None
DEFAULT_DATASET_ROOT = Path(r"K:\23所雷达数据\CQ-08中国航天科工二院二十三所-低空监视雷达目标智能识别技术研究数据集")
DEFAULT_TRAIN_REGISTRY = Path(r"K:\radar\main\data\manifests\rdx_train_registry.json")
DEFAULT_GROUPED_SPLIT = Path(r"K:\radar\main\data\manifests\cq08_grouped_split_f.json")
DEFAULT_F_SPLIT_RD_CACHE = Path(r"H:\RadarInsight\cache\rd_vr360_w900_db_linear_fsplit42_train32")
DEFAULT_TRACK_INDEX = Path(r"K:\radar\main\data\processed\expert1_track_index.csv")
DEFAULT_TR_CHECKPOINT = Path(
    r"K:\radar\main\artifacts\f_protocol\20260728-183147\b01_transformer_seed42\best_model.pt"
)
DEFAULT_RD_CHECKPOINT = (
    ROOT / "artifacts" / "rd_ablation_R2_contrast_w900_registry_seed42_rerun_rerun4_rerun" / "best.pt"
)
PARTITION_AUGMENTATION_DEFAULTS = {
    "partition_augmentation_diagnostics": True,
    "partition_augmentation_method": "perturbation",
    # Balance each fixed split to its largest original class. Validation and
    # test copies remain diagnostics and never affect optimization or model
    # selection.
    "partition_augmentation_targets_train": [489, 489, 489, 489, 489],
    "partition_augmentation_train_enabled": [True, True, True, True, True],
    "partition_augmentation_targets_val": [105, 105, 105, 105, 105],
    "partition_augmentation_targets_test": [105, 105, 105, 105, 105],
}
TR_LABEL_TO_INDEX = {
    "DroneTarget": 0, "BirdTarget": 1, "BalloonTarget": 2,
    "ClutterTarget": 3, "UnknownTarget": 4,
}
TR_TRAIN_COUNT_CACHE: dict[tuple[str, str], tuple[tuple[int, int, int, int], list[int]]] = {}


def tr_train_base_counts(split_path: Union[Path, str], track_index_path: Union[Path, str]) -> list[int]:
    """Return the fixed-split TR train counts without loading track CSVs.

    The UI uses this for a faithful preview of the same automatic CE weights
    the trainer derives after deterministic train-record expansion.
    """
    split_file = Path(split_path).expanduser().resolve()
    index_file = Path(track_index_path).expanduser().resolve()
    split_stat, index_stat = split_file.stat(), index_file.stat()
    stamp = (split_stat.st_mtime_ns, split_stat.st_size, index_stat.st_mtime_ns, index_stat.st_size)
    key = (str(split_file), str(index_file))
    cached = TR_TRAIN_COUNT_CACHE.get(key)
    if cached and cached[0] == stamp:
        return list(cached[1])
    split_payload = json.loads(split_file.read_text(encoding="utf-8-sig"))
    if "train_group_ids" in split_payload:
        train_ids = {str(value) for value in split_payload["train_group_ids"]}
    else:
        train_ids = {str(key) for key, value in split_payload.items() if value == "train"}
    counts = [0] * 5
    import csv
    with index_file.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("source_name", "")) != "cq08_track":
                continue
            sample_id = str(row.get("sample_id", ""))
            prefix = "cq08|track|"
            if not sample_id.startswith(prefix) or sample_id[len(prefix):] not in train_ids:
                continue
            label = TR_LABEL_TO_INDEX.get(str(row.get("label_name", "")))
            if label is not None:
                counts[label] += 1
    if not any(counts):
        raise ValueError("No CQ-08 train trajectories were found in the selected split/index")
    TR_TRAIN_COUNT_CACHE[key] = (stamp, counts)
    return list(counts)


def automatic_tr_class_loss_preview(
    split_path: Union[Path, str], track_index_path: Union[Path, str], targets, *, mode: str,
    beta: float, floor: float, cap: float,
) -> dict[str, object]:
    """Mirror ``select_class_weights`` without importing the training module."""
    import math
    if mode not in {"inverse_sqrt", "class_balanced"}:
        raise ValueError("Unknown automatic class-loss rule")
    if not 0.0 <= beta < 1.0 or floor < 0.0 or cap < floor:
        raise ValueError("Invalid automatic class-loss parameters")
    if not isinstance(targets, list) or len(targets) != 5:
        raise ValueError("Train targets require five values")
    target_values = [int(value) for value in targets]
    if any(value < 0 or value > 10000 for value in target_values):
        raise ValueError("Train targets must be between 0 and 10000")
    base_counts = tr_train_base_counts(split_path, track_index_path)
    effective_counts = [max(base, target) for base, target in zip(base_counts, target_values)]
    if mode == "inverse_sqrt":
        weights = [1.0 / math.sqrt(max(1, count)) for count in effective_counts]
    else:
        weights = [(1.0 - beta) / max(1e-12, 1.0 - beta ** max(1, count)) for count in effective_counts]
    mean = sum(weights) / len(weights) or 1.0
    normalized = [min(cap, max(floor, value / mean)) for value in weights]
    return {"base_counts": base_counts, "effective_counts": effective_counts,
            "raw_weights": weights, "weights": normalized,
            "rule": mode, "beta": beta, "floor": floor, "cap": cap}


def automatic_tr_sampling_preview(
    split_path: Union[Path, str], track_index_path: Union[Path, str], targets, *, mode: str,
) -> dict[str, object]:
    """Mirror the TR sampler and expose readable, relative per-class weights.

    WeightedRandomSampler is invariant to multiplying every record weight by a
    common constant.  The UI therefore reports the five per-record weights
    normalized to a five-class mean of 1.0, while also returning the actual
    resulting class probabilities.  The default Clutter multiplier remains
    the B01 protocol value of 2.0; the form's manual mode can override the
    five extra multipliers separately.
    """
    if mode not in {"b01_balanced", "inverse_frequency"}:
        raise ValueError("Unknown automatic sampling rule")
    if not isinstance(targets, list) or len(targets) != 5:
        raise ValueError("Train targets require five values")
    target_values = [int(value) for value in targets]
    if any(value < 0 or value > 10000 for value in target_values):
        raise ValueError("Train targets must be between 0 and 10000")
    base_counts = tr_train_base_counts(split_path, track_index_path)
    effective_counts = [max(base, target) for base, target in zip(base_counts, target_values)]
    if mode == "b01_balanced":
        max_count = float(max(effective_counts) or 1)
        base = [(max_count / max(count, 1)) ** 0.25 for count in effective_counts]
        rule = "B01 温和平衡：(最大类数 / 本类数)^0.25"
    else:
        base = [1.0 / max(count, 1) for count in effective_counts]
        rule = "类别逆频率：1 / 本类航迹数"
    # These are the protocol's automatic extra multipliers.  Their absolute
    # scale is immaterial to WeightedRandomSampler, so normalize only for a
    # readable preview without changing the training command.
    protocol_extra = [1.0, 1.0, 1.0, 2.0, 1.0]
    raw = [value * extra for value, extra in zip(base, protocol_extra)]
    mean = sum(raw) / 5.0 or 1.0
    relative = [value / mean for value in raw]
    masses = [count * value for count, value in zip(effective_counts, raw)]
    total = sum(masses) or 1.0
    probabilities = [mass / total for mass in masses]
    return {
        "base_counts": base_counts, "effective_counts": effective_counts,
        "base_weights": base, "relative_weights": relative,
        "class_sampling_probability": probabilities, "rule": rule,
        "protocol_extra": protocol_extra,
    }


def automatic_tr_balance_preview(
    split_path: Union[Path, str], track_index_path: Union[Path, str], targets,
    *, enabled=None, sampling_enabled=None, loss_enabled=None, sampling_mode: str, sampling_boosts=None,
    class_weight_mode: str, beta: float, floor: float, cap: float,
    manual_loss_weights=None,
) -> dict[str, object]:
    """Return one auditable view of all class-balance stages.

    ``combined_weight`` is a diagnostic product of sampler and loss weights;
    it is not a new training parameter.  The trainer still applies sampling
    and loss weighting independently.
    """
    import math
    base_counts = tr_train_base_counts(split_path, track_index_path)
    if not isinstance(targets, list) or len(targets) != 5:
        raise ValueError("Train targets require five values")
    target_values = [int(value) for value in targets]
    if any(value < 0 or value > 10000 for value in target_values):
        raise ValueError("Train targets must be between 0 and 10000")
    enabled_values = [True] * 5 if enabled is None else [bool(value) for value in enabled]
    if len(enabled_values) != 5:
        raise ValueError("Train augmentation switches require five values")
    sampling_enabled_values = [True] * 5 if sampling_enabled is None else [bool(value) for value in sampling_enabled]
    loss_enabled_values = [True] * 5 if loss_enabled is None else [bool(value) for value in loss_enabled]
    if len(sampling_enabled_values) != 5 or len(loss_enabled_values) != 5:
        raise ValueError("Class weighting switches require five values")
    effective_counts = [max(base, target) if enabled_values[index] else base
                        for index, (base, target) in enumerate(zip(base_counts, target_values))]
    count_mean = sum(effective_counts) / 5.0 or 1.0
    augmentation_count_weights = [count / count_mean for count in effective_counts]
    if sampling_mode not in {"b01_balanced", "inverse_frequency"}:
        raise ValueError("Unknown automatic sampling rule")
    max_count = float(max(effective_counts) or 1)
    sampling_base = [
        (max_count / max(count, 1)) ** 0.25 if sampling_mode == "b01_balanced"
        else 1.0 / max(count, 1)
        for count in effective_counts
    ]
    if sampling_boosts is None:
        boosts = [1.0, 1.0, 1.0, 2.0, 1.0]
    else:
        if not isinstance(sampling_boosts, list) or len(sampling_boosts) != 5:
            raise ValueError("Sampling boosts require five values")
        boosts = [float(value) for value in sampling_boosts]
        if any(value <= 0.0 for value in boosts):
            raise ValueError("Sampling boosts must be positive")
    sampling_raw = [base * boost for base, boost in zip(sampling_base, boosts)]
    sampling_mean = sum(sampling_raw) / 5.0 or 1.0
    sampling_weights = [value / sampling_mean for value in sampling_raw]
    masses = [count * value for count, value in zip(effective_counts, sampling_raw)]
    mass_total = sum(masses) or 1.0
    probabilities = [value / mass_total for value in masses]
    # `sampling_weights` is the per-record sampler multiplier.  For the UI
    # diagnostic we also expose the class-level effective weight, which
    # includes the number of records in each class and is normalized so the
    # five-class mean is 1.0.  This is proportional to the actual probability
    # that a class contributes a sampled record.
    probability_mean = sum(probabilities) / 5.0 or 1.0
    sampling_effective_weights = [value / probability_mean if sampling_enabled_values[index] else 1.0
                                  for index, value in enumerate(probabilities)]
    if class_weight_mode not in {"inverse_sqrt", "class_balanced"}:
        raise ValueError("Unknown automatic class-loss rule")
    if not 0.0 <= beta < 1.0 or floor < 0.0 or cap < floor:
        raise ValueError("Invalid automatic class-loss parameters")
    if manual_loss_weights is not None:
        if not isinstance(manual_loss_weights, list) or len(manual_loss_weights) != 5:
            raise ValueError("Manual class-loss weights require five values")
        raw_loss = [float(value) for value in manual_loss_weights]
        if any(value <= 0.0 or not math.isfinite(value) for value in raw_loss):
            raise ValueError("Manual class-loss weights must be positive finite values")
    else:
        if class_weight_mode == "inverse_sqrt":
            raw_loss = [1.0 / math.sqrt(max(1, count)) for count in effective_counts]
        else:
            raw_loss = [(1.0 - beta) / max(1e-12, 1.0 - beta ** max(1, count))
                        for count in effective_counts]
    class_loss_raw_weights = list(raw_loss)
    loss_mean = sum(raw_loss) / 5.0 or 1.0
    loss_normalized_weights = [value / loss_mean for value in raw_loss]
    loss_weights = [min(cap, max(floor, value)) for value in loss_normalized_weights]
    loss_weights = [value if loss_enabled_values[index] else 1.0
                    for index, value in enumerate(loss_weights)]
    # Expected class influence after both sampler and loss weighting.  Use the
    # class-level sampling contribution here so original class counts are not
    # silently discarded from the final diagnostic.
    combined_raw = [sample * loss for sample, loss in zip(sampling_effective_weights, loss_weights)]
    combined_mean = sum(combined_raw) / 5.0 or 1.0
    combined_weights = [value / combined_mean for value in combined_raw]
    return {
        "base_counts": base_counts,
        "requested_targets": target_values,
        "augmentation_enabled": enabled_values,
        "sampling_enabled": sampling_enabled_values,
        "loss_enabled": loss_enabled_values,
        "effective_counts": effective_counts,
        "augmentation_count_weights": augmentation_count_weights,
        "sampling_base_weights": sampling_base,
        "sampling_boosts": boosts,
        "sampling_weights": sampling_weights,
        "sampling_raw_weights": sampling_raw,
        "sampling_class_masses": masses,
        "sampling_effective_weights": sampling_effective_weights,
        "class_sampling_probability": probabilities,
        "class_loss_weights": loss_weights,
        "class_loss_raw_weights": class_loss_raw_weights,
        "class_loss_normalized_weights": loss_normalized_weights,
        "combined_raw_weights": combined_raw,
        "combined_mean": combined_mean,
        "combined_weights": combined_weights,
        "combined_weight_note": "联合权重仅用于诊断；训练仍分别使用采样倍率和类别损失权重。",
    }
EXPERIMENT_NAME_RE = re.compile(r"[\w\u4e00-\u9fff][\w\u4e00-\u9fff.-]{0,119}", re.UNICODE)
SCHEDULER = None
DISCOVER_CACHE_LOCK = threading.Lock()
DISCOVER_CACHE = {"time": 0.0, "value": None}
DISCOVER_CACHE_SECONDS = 30.0


def _output_index_key(name):
    """Return the canonical, persistent identity of an experiment output."""
    return f"output:{(ROOT / 'artifacts' / str(name)).resolve()}"


def _experiment_index_key(item):
    """Return one stable key for the *experiment*, not for a queue attempt.

    A queued run and its eventual artifact have different incidental values
    (queue id versus directory scan).  Using either of those as the primary
    identity made one run consume two numbers after it completed.  The output
    directory is chosen at submission time and stays with the run throughout
    its lifecycle, so it is the only suitable primary key here.
    """
    output_dir = item.get("output_dir")
    if output_dir:
        try:
            return f"output:{Path(output_dir).expanduser().resolve()}"
        except (OSError, ValueError, TypeError):
            pass
    name = item.get("name")
    if name:
        return _output_index_key(name)
    queue_id = item.get("queue_id")
    return f"queue:{queue_id}" if queue_id else "unknown:experiment"


def _load_experiment_index():
    try:
        value = json.loads(EXPERIMENT_INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("experiments"), dict):
            return value
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return {"version": 1, "next_number": 1, "experiments": {}}


def _save_experiment_index(value):
    EXPERIMENT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = EXPERIMENT_INDEX_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(EXPERIMENT_INDEX_PATH)


def _experiment_start_time(item, record=None):
    """Return the submission-time key used for chronological numbering.

    The displayed number is an experiment identifier, therefore it must not
    change merely because a queued run begins later.  ``created_at`` is the
    scheduler submission time; ``started_at`` is only the fallback for old
    artifacts that predate the scheduler.
    """
    record = record or {}
    for source in (item, record):
        for field in ("created_at", "started_at", "updated_at"):
            value = source.get(field)
            if value:
                return value
    return "9999-12-31T23:59:59"


def _experiment_sort_key(item, record=None):
    record = record or {}
    start_time = _experiment_start_time(item, record)
    queue_order = item.get("queue_order")
    if queue_order is None:
        queue_order = record.get("queue_order")
    try:
        queue_order = int(queue_order) if queue_order is not None else 2**31
    except (TypeError, ValueError):
        queue_order = 2**31
    return (start_time, queue_order, item.get("name", ""))


def _migrate_experiment_index(index, items):
    """Merge legacy queue/name records into one chronological experiment list.

    Versions 1--3 used ``queue:<id>`` while the scheduler was active and
    ``name:<name>`` after an artifact was scanned.  Consequently a single
    training run could receive two different numbers.  Version 4 rebuilds
    only from the currently discoverable experiments and gives each output
    directory exactly one number.  This is deliberately a one-time repair:
    later numbers are append-only and never depend on UI refresh order.
    """
    if int(index.get("version") or 1) >= 4:
        return False
    # A directory should appear once even while its scheduler record is also
    # present.  Keep the richest item (normally the artifact-backed one).
    unique = {}
    for item in items:
        key = _experiment_index_key(item)
        previous = unique.get(key)
        if previous is None or len(item.get("config") or {}) >= len(previous.get("config") or {}):
            unique[key] = item
    ordered = sorted(unique.items(), key=lambda pair: _experiment_sort_key(pair[1]))
    migrated = {}
    for number, (key, item) in enumerate(ordered, start=1):
        started_at = item.get("started_at")
        sort_time = _experiment_start_time(item)
        migrated[key] = {
            "number": number,
            "created_at": item.get("created_at") or sort_time,
            "started_at": started_at,
            "sort_time": sort_time,
            "queue_order": item.get("queue_order"),
        }
    index["experiments"] = migrated
    index["version"] = 4
    index["numbering"] = "created_at"
    index["next_number"] = len(migrated) + 1
    return True


def assign_experiment_numbers(items):
    """Assign stable numbers in experiment start order."""
    with EXPERIMENT_INDEX_LOCK:
        index = _load_experiment_index()
        records = index.setdefault("experiments", {})
        _migrate_experiment_index(index, items)
        records = index.setdefault("experiments", {})
        item_by_key = {_experiment_index_key(item): item for item in items}
        pending = []
        for item in items:
            key = _experiment_index_key(item)
            record = records.get(key)
            start_time = _experiment_start_time(item, record)
            if record is None:
                pending.append((_experiment_sort_key(item), item.get("name", ""), key,
                                item.get("started_at"), item.get("created_at")))
            else:
                item["experiment_number"] = record.get("number")
                item["experiment_created_at"] = record.get("started_at") or record.get("created_at") or start_time
                if item.get("started_at") and not record.get("started_at"):
                    record["started_at"] = item["started_at"]
                    record["sort_time"] = item["started_at"]
        pending.sort(key=lambda value: (value[0], value[1]))
        next_number = int(index.get("next_number") or 1)
        for sort_key, _, key, started_at, created_at in pending:
            sort_time = sort_key[0]
            pending_item = item_by_key.get(key) or {}
            records[key] = {"number": next_number, "created_at": created_at or sort_time,
                            "started_at": started_at, "sort_time": sort_time,
                            "queue_order": pending_item.get("queue_order")}
            next_number += 1
        index["next_number"] = next_number
        index["version"] = max(int(index.get("version") or 1), 4)
        index["numbering"] = "created_at"
        for item in items:
            record = records.get(_experiment_index_key(item))
            if record:
                item["experiment_number"] = record.get("number")
                item["experiment_created_at"] = record.get("started_at") or record.get("created_at") or item.get("started_at")
        _save_experiment_index(index)
    return items


def rename_experiment_index(old_name, new_name):
    with EXPERIMENT_INDEX_LOCK:
        index = _load_experiment_index()
        records = index.setdefault("experiments", {})
        old_key, new_key = _output_index_key(old_name), _output_index_key(new_name)
        if old_key in records and new_key not in records:
            records[new_key] = records.pop(old_key)
        # Kept only for indexes created before the v4 migration is first
        # reached (for example a rename immediately after a server upgrade).
        elif f"name:{old_name}" in records and new_key not in records:
            records[new_key] = records.pop(f"name:{old_name}")
        _save_experiment_index(index)


def is_training_command(command):
    # Include branch evaluation jobs because they share the same queue,
    # resource accounting, cancellation and monitor surface as RD training.
    return (
        "radar_rd.train" in command
        or "training_code_snapshots" in command
        or re.search(r"radar_rd[\\/]+train\.py(?:\s|\"|$)", command, re.IGNORECASE) is not None
        or re.search(r"scripts[\\/]+(?:train_tr_only|train_tr_ml|evaluate_tr_only|evaluate_rd_only|evaluate_soft_cascade|generate_fusion_oof|train_calibration_gate)\.py(?:\s|\"|$)", command, re.IGNORECASE) is not None
    )


def progress_positions(experiment):
    progress = experiment["progress"]
    config = experiment["config"]
    phase = progress.get("phase")
    batch = float(progress.get("batch") or 0)
    total_batches = float(progress.get("total_batches") or 0)
    phase_percent = 100.0 * batch / total_batches if total_batches > 0 else float(progress.get("stage_percent") or 0.0)
    explicit_overall = progress.get("overall_percent")
    if explicit_overall is not None:
        try:
            overall = float(explicit_overall)
        except (TypeError, ValueError):
            overall = None
        if overall is not None:
            return phase, phase_percent, max(0.0, min(100.0, overall))
    if str(config.get("experiment_type") or "").lower() in {"fusion_gate_training", "fusion_gate_calibration"}:
        overall = float(progress.get("percent") or 0.0)
        return phase, phase_percent, max(0.0, min(100.0, overall))
    epochs = int(progress.get("total_epochs") or config.get("epochs") or 0)
    epoch = int(progress.get("epoch") or 0)
    if not epochs or not epoch:
        return phase, phase_percent, None
    # Reserve 1% for final testing. Training and validation occupy 70% and
    # 30% of each epoch respectively; live timing samples correct the speed.
    epoch_weight = 99.0 / epochs
    if phase == "train":
        overall = (epoch - 1) * epoch_weight + epoch_weight * 0.70 * phase_percent / 100.0
    elif phase == "validation":
        overall = (epoch - 1) * epoch_weight + epoch_weight * (0.70 + 0.30 * phase_percent / 100.0)
    elif phase == "testing":
        overall = 99.0 + phase_percent * 0.01
    else:
        overall = len(experiment["history"]) * epoch_weight
    return phase, phase_percent, max(0.0, min(100.0, overall))


def estimate_remaining(samples, current):
    if len(samples) < 3:
        return None
    newest_time, newest_value = samples[-1]
    oldest_time, oldest_value = samples[0]
    elapsed = newest_time - oldest_time
    advanced = newest_value - oldest_value
    if elapsed < 6.0 or advanced <= 0.05:
        return None
    speed = advanced / elapsed
    remaining = (100.0 - current) / speed
    return remaining if 0 <= remaining <= 30 * 24 * 3600 else None


def add_eta(experiment):
    experiment["eta"] = {}
    # Keep the phase-independent position available for both live and
    # historical experiments.  The raw progress.percent is phase-local.
    _, _, overall_percent = progress_positions(experiment)
    experiment["overall_percent"] = overall_percent
    if not experiment["running"]:
        return experiment
    phase, phase_percent, overall_percent = progress_positions(experiment)
    now = time.time()
    key = experiment["output_dir"]
    phase_key = f"{phase}:{experiment['progress'].get('epoch')}"
    with ETA_LOCK:
        state = ETA_SAMPLES.setdefault(key, {"phase_key": phase_key, "phase": deque(maxlen=240), "overall": deque(maxlen=600)})
        if state["phase_key"] != phase_key:
            state["phase_key"] = phase_key
            state["phase"].clear()
        if not state["phase"] or phase_percent != state["phase"][-1][1]:
            state["phase"].append((now, phase_percent))
        if overall_percent is not None and (not state["overall"] or overall_percent != state["overall"][-1][1]):
            state["overall"].append((now, overall_percent))
        cutoff = now - 600
        while state["phase"] and state["phase"][0][0] < cutoff:
            state["phase"].popleft()
        while state["overall"] and state["overall"][0][0] < cutoff:
            state["overall"].popleft()
        phase_seconds = estimate_remaining(state["phase"], phase_percent)
        overall_seconds = estimate_remaining(state["overall"], overall_percent) if overall_percent is not None else None
    experiment["eta"] = {
        "phase_seconds": phase_seconds,
        "overall_seconds": overall_seconds,
        "phase_completion": datetime.fromtimestamp(now + phase_seconds).isoformat(timespec="seconds") if phase_seconds is not None else None,
        "overall_completion": datetime.fromtimestamp(now + overall_seconds).isoformat(timespec="seconds") if overall_seconds is not None else None,
        "overall_percent": overall_percent,
    }
    return experiment


def powershell_processes():
    # Enumerating from inside the monitor through PowerShell can block on
    # Windows while the monitor is itself servicing requests.  psutil is
    # already used for resource accounting and provides the same information
    # without spawning a nested shell.  Keep the PowerShell path as a fallback
    # for environments where psutil is unavailable or cannot inspect a process.
    if psutil is not None:
        processes = []
        try:
            for process in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    name = (process.info.get("name") or "").lower()
                    if name not in {"python.exe", "python"}:
                        continue
                    command_line = process.info.get("cmdline") or []
                    if isinstance(command_line, (list, tuple)):
                        command_line = " ".join(str(part) for part in command_line)
                    processes.append({"ProcessId": int(process.info["pid"]),
                                     "CommandLine": str(command_line)})
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, TypeError, ValueError):
                    continue
            if processes:
                return processes
        except (psutil.Error, OSError):
            pass
    command = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    try:
        raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", command], text=True, encoding="utf-8", errors="replace")
        value = json.loads(raw) if raw.strip() else []
        return value if isinstance(value, list) else [value]
    except Exception:
        return []


def nvidia_resources():
    total = {"available": False, "utilization": None, "memory_percent": None,
             "memory_used_mb": None, "memory_total_mb": None, "name": None}
    per_process = {}
    active_compute_pids = set()
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, encoding="utf-8", errors="replace", timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        row = next((line for line in raw.splitlines() if line.strip()), "")
        name, utilization, used, capacity = [part.strip() for part in row.split(",", 3)]
        used, capacity = float(used), float(capacity)
        total.update({"available": True, "name": name, "utilization": float(utilization),
                      "memory_used_mb": used, "memory_total_mb": capacity,
                      "memory_percent": 100.0 * used / capacity if capacity else None})
    except (OSError, subprocess.SubprocessError, ValueError):
        return total, per_process, active_compute_pids
    try:
        # WDDM often reports '-' for SM utilization in pmon, while the
        # compute-app query still exposes the CUDA process PID.  Keep those
        # PIDs as an activity signal for the per-training fallback below.
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, encoding="utf-8", errors="replace", timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for line in raw.splitlines():
            try:
                active_compute_pids.add(int(line.strip()))
            except ValueError:
                continue
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "pmon", "-s", "u", "-c", "1"],
            text=True, encoding="utf-8", errors="replace", timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for line in raw.splitlines():
            match = re.match(r"\s*\d+\s+(\d+)\s+\S+\s+([\d-]+)\s+([\d-]+)", line)
            if not match:
                continue
            pid, sm, memory = match.groups()
            per_process[int(pid)] = {
                "gpu_percent": float(sm) if sm != "-" else 0.0,
                "gpu_memory_activity": float(memory) if memory != "-" else 0.0,
            }
    except (OSError, subprocess.SubprocessError):
        pass
    return total, per_process, active_compute_pids


def resource_snapshot():
    global SYSTEM_CPU_CACHE
    now = time.monotonic()
    with RESOURCE_LOCK:
        if RESOURCE_CACHE["value"] is not None and now - RESOURCE_CACHE["time"] < 1.0:
            return RESOURCE_CACHE["value"]
        gpu, gpu_processes, active_compute_pids = nvidia_resources()
        if psutil is None:
            value = {"system": {"cpu_percent": None, "memory_percent": None,
                                  "memory_used_gb": None, "memory_total_gb": None, "gpu": gpu},
                     "trainings": {}, "sampled_at": datetime.now().isoformat(timespec="seconds")}
            RESOURCE_CACHE.update({"time": now, "value": value})
            return value
        memory = psutil.virtual_memory()
        cpu_times = psutil.cpu_times()
        cpu_total = sum(cpu_times)
        cpu_idle = float(cpu_times.idle + getattr(cpu_times, "iowait", 0.0))
        if SYSTEM_CPU_CACHE is None:
            system_cpu_percent = psutil.cpu_percent(interval=0.05)
        else:
            total_delta = cpu_total - SYSTEM_CPU_CACHE[0]
            idle_delta = cpu_idle - SYSTEM_CPU_CACHE[1]
            system_cpu_percent = 100.0 * (total_delta - idle_delta) / total_delta if total_delta > 0 else 0.0
        SYSTEM_CPU_CACHE = (cpu_total, cpu_idle)
        training_roots = {}
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = subprocess.list2cmdline(process.info.get("cmdline") or [])
                if is_training_command(command):
                    training_roots[process.pid] = process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        trainings = {}
        gpu_active_groups = []
        active_pids = set()
        cpu_count = max(psutil.cpu_count() or 1, 1)
        for root_pid, root in training_roots.items():
            try:
                group = [root] + root.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                group = [root]
            cpu_percent = 0.0
            memory_bytes = 0
            gpu_percent = 0.0
            gpu_active = False
            child_count = 0
            for process in group:
                try:
                    active_pids.add(process.pid)
                    cpu_times = process.cpu_times()
                    cpu_total = float(cpu_times.user + cpu_times.system)
                    previous = PROCESS_CPU_CACHE.get(process.pid)
                    if previous:
                        elapsed = now - previous[0]
                        if elapsed > 0:
                            cpu_percent += max(0.0, (cpu_total - previous[1]) / elapsed * 100.0 / cpu_count)
                    PROCESS_CPU_CACHE[process.pid] = (now, cpu_total)
                    # Sum the active working sets of the trainer and DataLoader
                    # workers. Windows' private allocation counter includes
                    # large reserved mappings and severely overstates usage.
                    memory_bytes += int(process.memory_info().rss)
                    gpu_percent += gpu_processes.get(process.pid, {}).get("gpu_percent", 0.0)
                    gpu_active = gpu_active or process.pid in active_compute_pids
                    if process.pid != root_pid:
                        child_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
            if gpu_active:
                gpu_active_groups.append((str(root_pid), gpu_percent))
            trainings[str(root_pid)] = {
                "cpu_percent": min(cpu_percent, 100.0),
                "memory_percent": 100.0 * memory_bytes / memory.total if memory.total else None,
                "memory_mb": memory_bytes / (1024 * 1024),
                "gpu_percent": min(gpu_percent, 100.0),
                "child_processes": child_count,
                "gpu_active": gpu_active,
            }
        # On WDDM, pmon can return '-' for a CUDA process even while the
        # aggregate GPU counter is non-zero.  Attribute the aggregate load
        # across active CUDA training roots instead of displaying a false 0%.
        missing_gpu_groups = [key for key, observed in gpu_active_groups if observed <= 0.0]
        if missing_gpu_groups:
            fallback = float(gpu.get("utilization") or 0.0) / max(len(gpu_active_groups), 1)
            for key in missing_gpu_groups:
                trainings[key]["gpu_percent"] = min(100.0, max(0.0, fallback))
        for pid in list(PROCESS_CPU_CACHE):
            if pid not in active_pids:
                PROCESS_CPU_CACHE.pop(pid, None)
        value = {
            "system": {"cpu_percent": max(0.0, min(100.0, system_cpu_percent)),
                       "memory_percent": memory.percent,
                       "memory_used_gb": (memory.total - memory.available) / (1024 ** 3),
                       "memory_total_gb": memory.total / (1024 ** 3), "gpu": gpu},
            "trainings": trainings,
            "sampled_at": datetime.now().isoformat(timespec="seconds"),
        }
        RESOURCE_CACHE.update({"time": now, "value": value})
        return value


def discover():
    now = time.monotonic()
    with DISCOVER_CACHE_LOCK:
        cached = DISCOVER_CACHE.get("value")
        cache_valid = cached is not None and now - float(DISCOVER_CACHE.get("time") or 0.0) < DISCOVER_CACHE_SECONDS
        has_live = bool(cached and any(item.get("status") in ACTIVE_STATES for item in cached))
    if cached is not None and has_live:
        return refresh_cached_live(cached)
    if cache_valid:
        return refresh_cached_live(cached)
    result = []
    known_outputs = set()
    for process in powershell_processes():
        command = process.get("CommandLine") or ""
        if not is_training_command(command):
            continue
        out_match = re.search(r"--output-dir\s+(?:\"([^\"]+)\"|(\S+))", command)
        if not out_match:
            continue
        output = Path(out_match.group(1) or out_match.group(2))
        if not output.is_absolute():
            output = ROOT / output
        output = output.resolve()
        log = output.parent / f"{output.name}.log"
        error_log = output.parent / f"{output.name}.err.log"
        result.append(add_eta(read_experiment(int(process["ProcessId"]), command, output, log, error_log, True)))
        known_outputs.add(output)
    artifacts = ROOT / "artifacts"
    if artifacts.exists():
        for output in artifacts.iterdir():
            if (not output.is_dir() or output in known_outputs or
                    not ((output / "config.json").exists() or (output / "metrics.json").exists())):
                continue
            log = output.parent / f"{output.name}.log"
            error_log = output.parent / f"{output.name}.err.log"
            result.append(read_experiment(None, "", output, log, error_log, False))
    if SCHEDULER is not None:
        known_names = {item["name"] for item in result}
        for queued in SCHEDULER.experiment_records():
            if queued["name"] in known_names:
                for item in result:
                    if item["name"] == queued["name"]:
                        item.update({key: value for key, value in queued.items() if value is not None})
                        queue_status = queued.get("queue_status")
                        if queue_status == CONFIRMATION_STATE:
                            item["status"] = queue_status
                            item["running"] = False
                        elif queue_status in ACTIVE_STATES or queue_status in TERMINAL_STATES:
                            # The scheduler owns process exit status. A Python
                            # warning written to stderr must not turn a valid
                            # completed evaluation into a frontend failure.
                            item["status"] = queue_status
                            if queue_status == "completed":
                                item["overall_percent"] = 100.0
                                if not queued.get("queue_error"):
                                    item["error"] = ""
                continue
            output = Path(queued["output_dir"])
            if output.exists() and (output / "config.json").exists():
                continue
            params = queued.get("params") or {}
            status = queued.get("queue_status") or "queued"
            result.append({
                "pid": queued.get("pid"), "command": "", "name": queued["name"],
                "output_dir": queued["output_dir"], "running": status in ACTIVE_STATES,
                "status": status, "config": params, "progress": {}, "history": [],
                "best": {}, "validation": {}, "test": {}, "error": queued.get("queue_error") or "",
                "ablation": {}, "overall_percent": 0.0, "started_at": queued.get("started_at"),
                "updated_at": queued.get("created_at"), "finished_at": queued.get("finished_at"),
                "elapsed_seconds": None, **queued,
            })
    # Queued scheduler records do not pass through read_experiment; normalize
    # them with the same contract before returning the API response.
    for item in result:
        # Scheduler metadata may arrive after read_experiment() (notably while
        # TR/fusion jobs are still preparing). Rebuild the public interface so
        # the frontend never temporarily misclassifies them as RD/unknown.
        item.update(experiment_interface(
            item.get("params") or item.get("config") or {},
            item.get("fusion") or {},
            item.get("validation") or {},
            item.get("test") or {},
            Path(item.get("output_dir", "")),
            item.get("gate") or {},
        ))
    # Keep the sidebar recency ordering, while experiment_number remains the
    # stable chronological identifier shown to the user.
    assign_experiment_numbers(result)
    result = sorted(result, key=lambda item: (item.get("updated_at") or "", item["name"]), reverse=True)
    with DISCOVER_CACHE_LOCK:
        DISCOVER_CACHE["time"] = time.monotonic()
        DISCOVER_CACHE["value"] = copy.deepcopy(result)
    return result


def invalidate_discover_cache():
    with DISCOVER_CACHE_LOCK:
        DISCOVER_CACHE["time"] = 0.0
        DISCOVER_CACHE["value"] = None


def checkpoint_catalog():
    """Expose best checkpoints from completed branch training runs.

    A fusion evaluation must consume the validation-selected best checkpoint,
    never a transient last-epoch resume checkpoint.  External B01/TR and the
    configured RD baseline remain available as explicit reference entries.
    """
    entries = []
    seen = set()
    # First collect genuine model-producing runs.  Evaluations may point at
    # the same file, but must never replace the source model's experiment
    # number in a picker.
    source_by_checkpoint = {}
    for item in discover():
        experiment_type = item.get("experiment_type")
        output = Path(item.get("output_dir", ""))
        if experiment_type not in {"tr_only", "rd_only"}:
            continue
        branch = "tr" if experiment_type == "tr_only" else "rd"
        checkpoint = (output / ("best_model.joblib" if str((item.get("config") or {}).get("tr_implementation", "dl")) == "ml" and branch == "tr" else "best.pt")).resolve()
        if checkpoint.is_file():
            source_by_checkpoint[(branch, str(checkpoint))] = item

    def add(branch, path, *, experiment, experiment_type, source, epoch=None, macro_f1=None, updated_at=None,
            experiment_number=None, dedupe_key=None, alias_of=None, compatible_tr_checkpoint=None,
            compatible_rd_checkpoint=None):
        candidate = Path(path).expanduser().resolve()
        key = dedupe_key or (branch, str(candidate))
        if key in seen or not candidate.is_file():
            return
        seen.add(key)
        entries.append({
            "branch": branch,
            "path": str(candidate),
            "experiment": experiment,
            "experiment_number": experiment_number,
            "experiment_type": experiment_type,
            "source": source,
            "best_epoch": epoch,
            "macro_f1": macro_f1,
            "updated_at": updated_at,
            "alias_of": alias_of,
            "compatible_tr_checkpoint": compatible_tr_checkpoint,
            "compatible_rd_checkpoint": compatible_rd_checkpoint,
        })

    for item in discover():
        experiment_type = item.get("experiment_type")
        output = Path(item.get("output_dir", ""))
        config = item.get("config") or {}
        validation = item.get("validation") or {}
        best = item.get("best") or {}
        epoch = validation.get("epoch") or best.get("epoch") or (item.get("ablation") or {}).get("best_epoch")
        macro_f1 = validation.get("macro_f1") or best.get("val_trajectory_macro_f1")
        common = {
            "experiment": item.get("name"), "experiment_type": experiment_type,
            "source": "training", "epoch": epoch, "macro_f1": macro_f1,
            "updated_at": item.get("updated_at"), "experiment_number": item.get("experiment_number"),
        }
        if experiment_type == "tr_only":
            checkpoint_name = "best_model.joblib" if str(config.get("tr_implementation", "dl")) == "ml" else "best.pt"
            add("tr", output / checkpoint_name, **common)
        elif experiment_type == "rd_only":
            add("rd", output / "best.pt", **common)
        elif experiment_type == "tr_checkpoint_eval":
            path = config.get("tr_checkpoint") or config.get("checkpoint", "")
            source_item = source_by_checkpoint.get(("tr", str(Path(path).expanduser().resolve())))
            # Keep the model-producing run as the canonical entry, but also
            # expose every completed evaluation as an alias when its config
            # points to a real checkpoint.  This lets users select the TR
            # checkpoint by the visible evaluation number (for example
            # 087 / Logos), even if source discovery cannot match the path.
            if Path(path).expanduser().is_file():
                add("tr", path, experiment=item.get("name"), experiment_type=experiment_type,
                    source="evaluation_reference", alias_of=(source_item or {}).get("name"),
                    epoch=config.get("checkpoint_epoch") or validation.get("epoch"),
                    macro_f1=validation.get("macro_f1"), updated_at=item.get("updated_at"),
                    experiment_number=item.get("experiment_number"),
                    dedupe_key=("tr-evaluation", str(Path(path).expanduser().resolve()), item.get("name")))
            else:
                add("tr", path, experiment=item.get("name"), experiment_type=experiment_type,
                    source="reference", epoch=config.get("checkpoint_epoch"),
                    macro_f1=validation.get("macro_f1"), updated_at=item.get("updated_at"),
                    experiment_number=item.get("experiment_number"))
        elif experiment_type == "rd_checkpoint_eval":
            path = config.get("rd_checkpoint") or config.get("checkpoint", "")
            source_item = source_by_checkpoint.get(("rd", str(Path(path).expanduser().resolve())))
            # Preserve the canonical RD training checkpoint and expose the
            # completed evaluation as a selectable alias as well.  This keeps
            # entries such as 052 / Entelechy visible in RD template pickers.
            if Path(path).expanduser().is_file():
                add("rd", path, experiment=item.get("name"), experiment_type=experiment_type,
                    source="evaluation_reference", alias_of=(source_item or {}).get("name"),
                    epoch=config.get("checkpoint_epoch") or validation.get("epoch"),
                    macro_f1=validation.get("macro_f1"), updated_at=item.get("updated_at"),
                    experiment_number=item.get("experiment_number"),
                    dedupe_key=("rd-evaluation", str(Path(path).expanduser().resolve()), item.get("name")))
            else:
                add("rd", path, experiment=item.get("name"), experiment_type=experiment_type,
                    source="reference", epoch=config.get("checkpoint_epoch"),
                    macro_f1=validation.get("macro_f1"), updated_at=item.get("updated_at"),
                    experiment_number=item.get("experiment_number"))
        elif experiment_type in {"fusion_gate_training", "fusion_gate_calibration"}:
            add("gate", output / "fusion_gate.pt", experiment=item.get("name"),
                experiment_type=experiment_type, source="calibration" if experiment_type == "fusion_gate_calibration" else "oof_training",
                macro_f1=((item.get("gate") or {}).get("gate_fit") or {}).get("macro_f1"),
                updated_at=item.get("updated_at"), experiment_number=item.get("experiment_number"),
                compatible_tr_checkpoint=config.get("tr_checkpoint"),
                compatible_rd_checkpoint=config.get("rd_checkpoint"))

    add("tr", DEFAULT_TR_CHECKPOINT, experiment="B01 外部基线", experiment_type="reference",
        source="reference", updated_at=None)
    add("rd", DEFAULT_RD_CHECKPOINT, experiment="默认 RD 基线", experiment_type="reference",
        source="reference", updated_at=None)
    return sorted(entries, key=lambda entry: (entry["branch"], entry.get("updated_at") or "", entry["experiment"] or ""), reverse=True)


def refresh_cached_live(cached):
    """Refresh only queued/running jobs without rescanning all artifacts."""
    live = copy.deepcopy(cached)
    scheduler_records = {
        record.get("name"): record
        for record in (SCHEDULER.experiment_records() if SCHEDULER is not None else [])
    }
    for item in live:
        name = item.get("name")
        queued = scheduler_records.get(name)
        if queued is not None:
            item.update({key: value for key, value in queued.items() if value is not None})
            queue_status = queued.get("queue_status")
            if queue_status == CONFIRMATION_STATE:
                item["status"] = queue_status
                item["running"] = False
            elif queue_status in ACTIVE_STATES or queue_status in TERMINAL_STATES:
                item["status"] = queue_status
                item["running"] = queue_status in ACTIVE_STATES
                if queue_status == "completed":
                    item["overall_percent"] = 100.0
                    if not queued.get("queue_error"):
                        item["error"] = ""
        output = Path(item.get("output_dir", ""))
        if not output.is_dir() or not (output / "config.json").is_file():
            continue

        # A scheduler record is deliberately lightweight: while a job is
        # running it does not contain metrics or the final progress snapshot.
        # Previously, when the scheduler changed a cached item from running to
        # completed, the code above changed only its status and then skipped
        # this disk refresh.  That left completed fusion evaluations with an
        # empty ``fusion`` object (and their old inference progress) until the
        # 30-second cache expired.  Refresh exactly those terminal items whose
        # persisted result is not represented by the cached API object.
        actual_progress = read_json(output / "progress.json") or {}
        terminal_cache_stale = bool(
            item.get("status") in TERMINAL_STATES and (
                ((output / "metrics.json").is_file() and not item.get("fusion"))
                or ((output / "test_trajectory_metrics.json").is_file() and not item.get("test"))
                # Partition-local augmented diagnostics can be generated after
                # the original checkpoint result.  Refresh the cached public
                # record as soon as those files appear; otherwise the browser
                # cannot expose the original/augmented view switch until the
                # full discovery TTL expires.
                or ((output / "validation_augmented_best.json").is_file()
                    and not item.get("validation_augmented"))
                or ((output / "test_augmented_trajectory_metrics.json").is_file()
                    and not item.get("test_augmented"))
                or ((output / "augmented_metrics.json").is_file()
                    and not item.get("augmented_fusion"))
                or ((output / "trajectory_decisions_augmented.jsonl").is_file()
                    and not (item.get("decisions") or {}).get("augmented_saved"))
                or ((output / "trajectory_decisions_test_augmented.jsonl").is_file()
                    and not (item.get("decisions") or {}).get("augmented_saved"))
                or ((output / "gate_metrics.json").is_file() and not item.get("gate"))
                or ((output / "ablation_complete.json").is_file() and not item.get("ablation"))
                or (isinstance(actual_progress, dict)
                    and str(actual_progress.get("phase", "")).lower() in {"complete", "completed", "done"}
                    and actual_progress != (item.get("progress") or {}))
            )
        )
        if item.get("status") not in ACTIVE_STATES and not terminal_cache_stale:
            continue
        pid = item.get("pid")
        alive = bool(pid and psutil is not None and psutil.pid_exists(int(pid)))
        try:
            refreshed = read_experiment(
                pid if alive else None, item.get("command", ""), output,
                output.parent / f"{output.name}.log",
                output.parent / f"{output.name}.err.log", alive,
            )
            item.update(refreshed)
            if not alive and (output / "ablation_complete.json").is_file():
                item["status"] = "completed"
                item["running"] = False
                item["overall_percent"] = 100.0
        except (OSError, ValueError, TypeError):
            continue
    # A newly queued job can create config.json before the next full discovery
    # pass.  Add scheduler records that are not yet represented in the cache;
    # otherwise the UI can miss a just-created experiment until the cache TTL.
    known_names = {str(item.get("name")) for item in live}
    for queued in scheduler_records.values():
        name = str(queued.get("name") or "")
        output = Path(str(queued.get("output_dir") or ""))
        # Scheduler records are authoritative even before the worker creates
        # the output directory/config.json.  Filtering those records by
        # output.is_dir() made a newly queued job disappear on cached polls.
        if not name or name in known_names:
            continue
        if (output / "config.json").is_file() or queued.get("queue_status") in ACTIVE_STATES | TERMINAL_STATES | {CONFIRMATION_STATE}:
            status = queued.get("queue_status") or "queued"
            live.append({
                "pid": queued.get("pid"), "command": "", "name": name,
                "output_dir": str(output), "running": status in ACTIVE_STATES,
                "status": status, "config": queued.get("params") or {},
                "progress": {}, "history": [], "best": {}, "validation": {},
                "test": {}, "error": queued.get("queue_error") or "",
                "ablation": {}, "overall_percent": 100.0 if status == "completed" else 0.0,
                "started_at": queued.get("started_at"), "updated_at": queued.get("created_at"),
                "finished_at": queued.get("finished_at"), "elapsed_seconds": None,
                **queued,
            })
            known_names.add(name)
    # Cached refreshes must use the same public interface normalization as a
    # full discovery. A live TR worker writes ``tr_only_training`` to its
    # config while the scheduler payload uses ``tr_only``; without this pass
    # the frontend can briefly fall back to its RD default label.
    for item in live:
        item.update(experiment_interface(
            item.get("params") or item.get("config") or {},
            item.get("fusion") or {},
            item.get("validation") or {},
            item.get("test") or {},
            Path(item.get("output_dir", "")),
            item.get("gate") or {},
        ))
    # Cached refreshes must use the same stable numbering path as a full
    # discovery.  Otherwise a new queue item alternated between ``--`` and its
    # persistent number while its output directory was being created.
    assign_experiment_numbers(live)
    with DISCOVER_CACHE_LOCK:
        DISCOVER_CACHE["time"] = time.monotonic()
        DISCOVER_CACHE["value"] = copy.deepcopy(live)
    return live


def read_json(path):
    # Training writers publish frequent snapshots.  Retry a transient decode
    # failure so legacy/direct writers cannot turn one poll into a visible
    # progress reset.
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            if attempt < 2:
                time.sleep(0.01)
    return None


def validate_rd_cache(cache_dir, params):
    """Return an incompatibility explanation, or ``None`` for a usable cache."""
    metadata = read_json(cache_dir / "metadata.json")
    complete = read_json(cache_dir / "complete.json")
    index = read_json(cache_dir / "index.json")
    if not isinstance(metadata, dict) or not isinstance(complete, dict) or not isinstance(index, dict):
        return "RD cache is incomplete"
    if complete.get("status") != "complete":
        return "RD cache has not finished building"
    frame_count = complete.get("frame_count")
    positions = []
    try:
        positions = sorted(int(value) for value in index.values())
    except (TypeError, ValueError):
        return "RD cache index has invalid positions"
    if (not isinstance(frame_count, int) or frame_count != len(index) or
            positions != list(range(len(index)))):
        return "RD cache index does not match its completed arrays"
    expected_preprocessing = {
        "velocity_min": float(params["velocity_min"]), "velocity_max": float(params["velocity_max"]),
        "target_width": int(params["target_width"]), "resampling": str(params["resampling"]),
    }
    if metadata.get("preprocessing") != expected_preprocessing:
        return "RD cache preprocessing does not match the current settings"
    if params["split_mode"] == "fixed_grouped":
        manifest = Path(params["grouped_split"]).resolve()
        expected_identity = {
            "split": {"mode": "fixed_grouped", "manifest": str(manifest),
                      "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "trajectory_count": 1549},
            "max_train_frames_per_trajectory": int(params["max_train_frames_per_trajectory"]),
            "include_test": False,
        }
        if metadata.get("cache_identity") != expected_identity:
            return "RD cache belongs to a different split or training-frame sampling policy"
    return None


def recommended_rd_cache(params):
    """Return the cache selected solely from the active RD training protocol.

    The browser must never have to repair a cache path inherited from a clone
    or local form state.  At present only the complete F/900/train32 cache has
    the identity metadata required by the strict RD training loader.  Other
    settings deliberately fall back to on-demand preprocessing until their
    own cache is built.
    """
    expected_f_protocol = (
        params.get("split_mode") == "fixed_grouped"
        and Path(str(params.get("grouped_split", ""))).resolve() == DEFAULT_GROUPED_SPLIT.resolve()
        and int(params.get("max_train_frames_per_trajectory", 0)) == 32
        and int(params.get("target_width", 0)) == 900
        and float(params.get("velocity_min", 0)) == -90.0
        and float(params.get("velocity_max", 0)) == 89.0
        and str(params.get("resampling", "")) == "db_linear"
    )
    if expected_f_protocol and validate_rd_cache(DEFAULT_F_SPLIT_RD_CACHE, params) is None:
        return str(DEFAULT_F_SPLIT_RD_CACHE.resolve())
    return ""


def rd_cache_build_status(cache_dir):
    """Small, safe status summary for the new-experiment form."""
    if not cache_dir.is_dir():
        return {"state": "missing"}
    complete = read_json(cache_dir / "complete.json")
    if isinstance(complete, dict) and complete.get("status") == "complete":
        return {"state": "ready", "frame_count": complete.get("frame_count")}
    state = read_json(cache_dir / "state.json")
    if isinstance(state, dict) and state.get("total"):
        processed, total = int(state.get("processed", 0)), int(state["total"])
        return {"state": "building", "processed": processed, "total": total,
                "percent": round(100.0 * processed / max(total, 1), 1)}
    return {"state": "initializing"}


def read_trajectory_decisions(output, branch, errors_only=False, offset=0, limit=100, source="original"):
    """Return a normalized, paged view of saved per-branch decisions."""
    fields = {
        "tr": ("tr_prediction", "tr_prediction_label", "tr_probabilities"),
        "rd": ("rd_prediction", "rd_prediction_label", "rd_probabilities"),
        "fusion": ("fused_prediction", "fused_prediction_label", "fused_probabilities"),
    }
    if branch not in fields:
        raise ValueError("branch must be tr, rd, or fusion")
    source = str(source or "original").lower()
    if source not in {"original", "augmented"}:
        raise ValueError("source must be original or augmented")
    if source == "original":
        candidates = [output / "trajectory_decisions.jsonl", output / "trajectory_decisions_test.jsonl"]
    else:
        candidates = [output / "trajectory_decisions_augmented.jsonl",
                      output / "trajectory_decisions_test_augmented.jsonl"]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        return {"items": [], "total": 0, "offset": offset, "limit": limit, "branch": branch, "source": source}
    prediction_field, label_field, probability_field = fields[branch]
    manifest_by_id = {}
    manifest_path = output / "partition_augmentation_manifest.json"
    manifest = read_json(manifest_path) if source == "augmented" else None
    if isinstance(manifest, dict):
        rows = manifest.get("records") or manifest.get("partition", {}).get("records") or []
        if not rows:
            for section in ("tr", "rd", "train", "val", "test"):
                value = manifest.get(section)
                if isinstance(value, dict) and isinstance(value.get("records"), list):
                    rows.extend(value["records"])
        if isinstance(rows, list):
            manifest_by_id = {str(row.get("trajectory_id")): row for row in rows if isinstance(row, dict)}
    items = []
    total = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            # TR-only artifacts predate branch-prefixed prediction fields.
            prediction = record.get(prediction_field)
            prediction_label = record.get(label_field)
            probabilities = record.get(probability_field)
            if branch == "tr" and prediction is None:
                prediction = record.get("prediction")
                prediction_label = record.get("prediction_label")
                probabilities = record.get("tr_probabilities")
            if prediction is None:
                continue
            truth = record.get("true_class")
            correct = int(prediction) == int(truth) if truth is not None else None
            if errors_only and correct:
                continue
            if total >= offset and len(items) < limit:
                confidence = None
                if isinstance(probabilities, list) and probabilities:
                    try:
                        confidence = float(max(probabilities))
                    except (TypeError, ValueError):
                        confidence = None
                trajectory_id = str(record.get("trajectory_id", ""))
                provenance = manifest_by_id.get(trajectory_id, {})
                is_virtual = bool("|aug-" in trajectory_id or "|smote-" in trajectory_id
                                  or record.get("augmentation_kind") or provenance)
                items.append({
                    "trajectory_id": trajectory_id,
                    "source_trajectory_id": (str(record.get("source_trajectory_id") or
                                                  record.get("augmentation_source_trajectory_id") or
                                                  provenance.get("source_trajectory_id") or
                                                  re.split(r"\|(?:aug|smote)-", trajectory_id, maxsplit=1)[0]) if is_virtual else None),
                    "is_virtual": is_virtual,
                    "augmentation_seed": (record.get("augmentation_seed") or provenance.get("seed")) if is_virtual else None,
                    "augmentation_method": ("smote" if provenance.get("source_trajectory_id_b")
                                             or "|smote-" in trajectory_id else "perturbation") if is_virtual else None,
                    "source_trajectory_id_b": provenance.get("source_trajectory_id_b") if is_virtual else None,
                    "interpolation_alpha": provenance.get("interpolation_alpha") if is_virtual else None,
                    "neighbor_candidates": provenance.get("neighbor_candidates") if is_virtual else None,
                    "copy_index": (record.get("copy_index") or provenance.get("copy_index")) if is_virtual else None,
                    "partition": (record.get("partition") or provenance.get("partition")) if is_virtual else None,
                    "true_class": truth,
                    "true_label": record.get("true_label"),
                    "prediction": prediction,
                    "prediction_label": prediction_label,
                    "confidence": confidence,
                    "correct": correct,
                    "branch_agreement": record.get("branch_agreement"),
                    "rd_frame_count": record.get("rd_frame_count") if branch == "rd" else None,
                    "rd_consistency": record.get("rd_consistency") if branch == "rd" else None,
                    "fusion_rescue_vs_tr": record.get("fusion_rescue_vs_tr") if branch == "fusion" else None,
                    "fusion_harm_vs_tr": record.get("fusion_harm_vs_tr") if branch == "fusion" else None,
                })
            total += 1
    return {"items": items, "total": total, "offset": offset, "limit": limit, "branch": branch, "source": source}


def experiment_interface(config, fusion_metrics, validation_metrics, test_metrics, output, gate_metrics=None):
    """Normalize heterogeneous artifacts for the frontend contract."""
    config = config or {}
    provenance = (fusion_metrics or {}).get("provenance") or {}
    explicit_type = str(config.get("experiment_type") or "").strip().lower()
    if explicit_type in {"fusion_gate_training", "fusion_gate_calibration"}:
        experiment_type = explicit_type
        metrics_source = "calibration_gate_fit" if explicit_type == "fusion_gate_calibration" else "oof_gate_fit"
        partition = config.get("calibration_partition") if explicit_type == "fusion_gate_calibration" else "train_oof"
    elif fusion_metrics or explicit_type in {"tr_rd_soft_cascade", "soft_cascade"}:
        experiment_type = "tr_rd_soft_cascade"
        metrics_source = "fusion.soft_cascade"
        partition = provenance.get("partition") or config.get("partition") or "val"
    elif explicit_type in {"tr_only", "tr_only_training", "trajectory_training"}:
        experiment_type = "tr_only"
        metrics_source = "test" if test_metrics else ("validation_best" if validation_metrics else None)
        partition = config.get("partition") or "val"
    elif explicit_type in {"tr_checkpoint_eval", "tr_only_reproduction", "trajectory_only"}:
        experiment_type = "tr_checkpoint_eval"
        metrics_source = "test" if test_metrics else ("validation_best" if validation_metrics else None)
        partition = config.get("partition") or "val"
    elif explicit_type in {"rd_checkpoint_eval", "rd_only_evaluation"}:
        experiment_type = "rd_checkpoint_eval"
        metrics_source = "test" if test_metrics else ("validation_best" if validation_metrics else None)
        partition = config.get("partition") or "val"
    elif (explicit_type in {"rd_only", "rd_training"} or config.get("input_mode")
          or config.get("target_width") or config.get("velocity_preprocessing")
          or config.get("normalization_mean") is not None
          or config.get("all_frame_counts") or config.get("max_train_frames_per_trajectory")):
        experiment_type = "rd_only"
        metrics_source = "test" if test_metrics else ("validation_best" if validation_metrics else None)
        split_config = config.get("split") if isinstance(config.get("split"), dict) else {}
        partition = config.get("partition") or split_config.get("partition") or "val"
    else:
        experiment_type = "unknown"
        metrics_source = None
        partition = config.get("partition") or "val"
    branch_metrics = {
        "tr": bool(experiment_type in {"tr_only", "tr_checkpoint_eval", "tr_rd_soft_cascade", "fusion_gate_training", "fusion_gate_calibration"}),
        "rd": bool(experiment_type in {"rd_only", "rd_checkpoint_eval", "tr_rd_soft_cascade", "fusion_gate_training", "fusion_gate_calibration"}),
        "fusion": bool(experiment_type in {"tr_rd_soft_cascade", "fusion_gate_training", "fusion_gate_calibration"}),
    }
    evaluation_metrics = test_metrics or validation_metrics or {}
    if experiment_type in {"fusion_gate_training", "fusion_gate_calibration"}:
        branch_outputs = {
            "tr": (gate_metrics or {}).get("tr_branch") or {},
            "rd": (gate_metrics or {}).get("rd_branch") or {},
            "fusion": (gate_metrics or {}).get("gate_fit") or {},
        }
        primary_branch = "fusion"
    elif experiment_type == "tr_rd_soft_cascade":
        branch_outputs = {
            "tr": (fusion_metrics or {}).get("tr_branch") or {},
            "rd": (fusion_metrics or {}).get("rd_branch") or {},
            "fusion": (fusion_metrics or {}).get("soft_cascade") or {},
        }
        primary_branch = "fusion"
    elif experiment_type in {"tr_only", "tr_checkpoint_eval"}:
        branch_outputs = {"tr": evaluation_metrics}
        primary_branch = "tr"
    elif experiment_type in {"rd_only", "rd_checkpoint_eval"}:
        branch_outputs = {"rd": evaluation_metrics}
        primary_branch = "rd"
    else:
        branch_outputs = {}
        primary_branch = None
    files = [name for name in (
        "config.json", "validation_best.json", "validation_latest.json",
        "test_trajectory_metrics.json", "metrics.json", "trajectory_decisions.jsonl",
        "trajectory_decisions_augmented.jsonl", "trajectory_decisions_test.jsonl",
        "trajectory_decisions_test_augmented.jsonl", "test_augmented_trajectory_metrics.json",
        "validation_augmented_best.json", "validation_augmented_latest.json",
        "augmented_metrics.json", "partition_augmentation_manifest.json",
        "augmentation_supplement.json",
        "best.pt", "best_model.pt", "best_model.joblib", "model_metadata.json", "candidate_metrics.json",
        "feature_importance.json", "last.pt", "ablation_complete.json",
        "fusion_gate.pt", "gate_metrics.json", "gate_history.json",
        "oof_predictions.jsonl", "oof_metadata.json", "fold_manifest.json",
        "calibration_scores.jsonl", "calibration_metadata.json",
    ) if (output / name).is_file()]
    return {
        "experiment_type": experiment_type,
        "experiment_label": config.get("experiment_label") or config.get("experiment_type") or experiment_type,
        "evaluation": {
            "partition": partition,
            "metrics_source": metrics_source,
            "primary_metric": "macro_f1",
            "test_allowed": bool(test_metrics),
            "branches": branch_metrics,
            "primary_branch": primary_branch,
        },
        "primary_branch": primary_branch,
        "branch_outputs": branch_outputs,
        "branches": {
            "tr": {"enabled": branch_metrics["tr"], "role": "trajectory"},
            "rd": {"enabled": branch_metrics["rd"], "role": "range_doppler"},
            "fusion": {"enabled": branch_metrics["fusion"], "role": "trajectory_level_soft_cascade"},
        },
        "outputs": files,
        "decisions": {
            "saved": (output / "trajectory_decisions.jsonl").is_file(),
            "augmented_saved": any((output / name).is_file() for name in (
                "trajectory_decisions_augmented.jsonl",
                "trajectory_decisions_test_augmented.jsonl",
            )),
            "artifact": "trajectory_decisions.jsonl" if (output / "trajectory_decisions.jsonl").is_file() else None,
            "branches": [key for key, value in branch_outputs.items() if value],
        },
    }


def read_experiment(pid, command, output, log, error_log, running):
    if not error_log.exists() and (output / "train.err.log").exists():
        error_log = output / "train.err.log"
    config = read_json(output / "config.json") or {}
    progress = read_json(output / "progress.json") or {}
    history = read_json(output / "history.json") or []
    validation_metrics = read_json(output / "validation_best.json") or {}
    validation_latest_metrics = read_json(output / "validation_latest.json") or {}
    validation_augmented_metrics = read_json(output / "validation_augmented_best.json") or {}
    validation_augmented_latest_metrics = read_json(output / "validation_augmented_latest.json") or {}
    test_metrics = read_json(output / "test_trajectory_metrics.json") or {}
    test_augmented_metrics = read_json(output / "test_augmented_trajectory_metrics.json") or {}
    ablation_metrics = read_json(output / "ablation_complete.json") or {}
    fusion_metrics = read_json(output / "metrics.json") or {}
    augmented_fusion_metrics = read_json(output / "augmented_metrics.json") or {}
    gate_metrics = read_json(output / "gate_metrics.json") or {}
    candidate_metrics = read_json(output / "candidate_metrics.json") or {}
    feature_importance = read_json(output / "feature_importance.json") or []
    if fusion_metrics and not config:
        provenance = fusion_metrics.get("provenance") or {}
        config = {
            "experiment_type": "tr_rd_soft_cascade",
            "experiment_label": "TR-RD 航迹级门控融合",
            "partition": provenance.get("partition", "val"),
            "fusion_mode": provenance.get("fusion_mode", "fixed"),
            "fixed_rd_weight": provenance.get("fixed_rd_weight"),
            "tr_checkpoint": provenance.get("tr_checkpoint"),
            "rd_checkpoint": provenance.get("rd_checkpoint"),
            "split": provenance.get("grouped_split"),
        }
    # For finished runs, progress.json may contain the final train/test phase
    # rather than the best validation checkpoint. Expose the checkpoint values
    # through progress as well so the monitor's metric cards cannot fall back
    # to a later, worse history row (for example 80.8% instead of 85.0%).
    if validation_metrics and not running:
        progress = dict(progress)
        progress["val_loss"] = validation_metrics.get("loss")
        progress["val_trajectory_macro_f1"] = validation_metrics.get("macro_f1")
        progress["val_trajectory_accuracy"] = validation_metrics.get("accuracy")
    error = error_log.read_text(encoding="utf-8", errors="replace")[-4000:] if error_log.exists() else ""

    # stderr is also used by PyTorch for harmless runtime warnings.  A
    # historical run is complete when it has a structurally valid terminal
    # artifact; only an actual traceback/exception should make it failed.
    terminal_artifact = bool(
        (isinstance(test_metrics, dict) and test_metrics)
        or ((output / "best_model.joblib").is_file() and isinstance(validation_metrics, dict) and validation_metrics)
        or (isinstance(ablation_metrics, dict) and ablation_metrics)
        or (isinstance(fusion_metrics, dict) and fusion_metrics)
        or (isinstance(gate_metrics, dict) and gate_metrics)
    )
    if isinstance(progress, dict) and str(progress.get("phase", "")).lower() in {"complete", "completed", "done"}:
        terminal_artifact = True
    real_error = bool(
        re.search(r"Traceback \(most recent call last\)", error, re.IGNORECASE)
        or re.search(r"(?:^|\n)\s*[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):", error)
        or re.search(r"RuntimeError|AssertionError|CalledProcessError|failed with exit code", error, re.IGNORECASE)
    )
    if running:
        status = "running"
    elif terminal_artifact:
        status = "completed"
    elif real_error:
        status = "failed"
    else:
        status = "incomplete"
    public_error = error if real_error else ""
    best = max(history, key=lambda item: item.get("val_trajectory_macro_f1", float("-inf")), default={})
    timestamp_paths = [output / "history.json", output / "progress.json", error_log]
    if test_metrics:
        timestamp_paths.append(output / "test_trajectory_metrics.json")
    if validation_metrics:
        timestamp_paths.append(output / "validation_best.json")
    if validation_latest_metrics:
        timestamp_paths.append(output / "validation_latest.json")
    if validation_augmented_metrics:
        timestamp_paths.append(output / "validation_augmented_best.json")
    if test_augmented_metrics:
        timestamp_paths.append(output / "test_augmented_trajectory_metrics.json")
    if augmented_fusion_metrics:
        timestamp_paths.append(output / "augmented_metrics.json")
    if fusion_metrics:
        timestamp_paths.append(output / "metrics.json")
    if gate_metrics:
        timestamp_paths.append(output / "gate_metrics.json")
    timestamps = [path.stat().st_mtime for path in timestamp_paths if path.exists()]
    updated_at = datetime.fromtimestamp(max(timestamps)).isoformat(timespec="seconds") if timestamps else None
    finished_at = updated_at if status in {"completed", "failed", "incomplete"} else None
    started_timestamp = output.stat().st_ctime if output.exists() else None
    if running and pid and psutil is not None:
        try:
            started_timestamp = psutil.Process(pid).create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
    started_at = datetime.fromtimestamp(started_timestamp).isoformat(timespec="seconds") if started_timestamp else None
    end_timestamp = time.time() if running else (max(timestamps) if timestamps else None)
    elapsed_seconds = max(0.0, end_timestamp - started_timestamp) if started_timestamp and end_timestamp else None
    overall_percent = progress_positions({"progress": progress, "config": config, "history": history})[2]
    if status == "completed":
        overall_percent = 100.0
    last_checkpoint = output / "last.pt"
    best_checkpoint = output / "best.pt"
    resume_checkpoint = last_checkpoint if last_checkpoint.is_file() else best_checkpoint
    interface = experiment_interface(config, fusion_metrics, validation_metrics, test_metrics, output, gate_metrics)
    return {"pid": pid, "command": command, "name": output.name, "output_dir": str(output),
            "running": running, "status": status, "config": config, "progress": progress,
            "history": history, "best": best, "validation": validation_metrics,
            "validation_latest": validation_latest_metrics,
            "validation_augmented": validation_augmented_metrics,
            "validation_augmented_latest": validation_augmented_latest_metrics,
            "test": test_metrics, "error": public_error, "ablation": ablation_metrics, "fusion": fusion_metrics,
            "test_augmented": test_augmented_metrics, "augmented_fusion": augmented_fusion_metrics,
            "gate": gate_metrics,
            "candidate_metrics": candidate_metrics, "feature_importance": feature_importance,
            "overall_percent": overall_percent,
            **interface,
            "created_at": started_at, "started_at": started_at, "updated_at": updated_at, "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "resume_available": status in {"failed", "incomplete", "cancelled"} and resume_checkpoint.is_file(),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint.is_file() else None,
            "resume_mode": "exact_last_epoch" if last_checkpoint.is_file() else ("legacy_best_warm_start" if best_checkpoint.is_file() else None)}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        request_path = parsed.path
        if request_path == "/api/scheduler":
            self.send_json(200, SCHEDULER.snapshot())
            return
        if request_path == "/api/resources":
            self.send_json(200, resource_snapshot())
            return
        if request_path == "/api/experiment-defaults":
            self.send_json(200, {
                "experiment_type": "rd_only",
                "dataset_root": str(DEFAULT_DATASET_ROOT), "epochs": 50,
                "batch_size": 128, "workers": 4,
                "max_train_frames_per_trajectory": 32, "norm_samples": 2048,
                "learning_rate": 0.0003, "weight_decay": 0.0001,
                "patience": 10, "seed": 42,
                "velocity_min": -90, "velocity_max": 89, "target_width": 900,
                "resampling": "db_linear", "normalization": "global_z",
                "input_mode": "rd", "model_head": "global", "augmentation": "off", "split_mode": "fixed_grouped",
                "train_registry": "", "rd_cache": str(DEFAULT_F_SPLIT_RD_CACHE),
                "rd_cache_status": rd_cache_build_status(DEFAULT_F_SPLIT_RD_CACHE), "skip_test": True,
                "grouped_split": str(DEFAULT_GROUPED_SPLIT),
                "track_index": str(DEFAULT_TRACK_INDEX),
                "tr_checkpoint": str(DEFAULT_TR_CHECKPOINT),
                "tr_implementation": "dl",
                "ml_models": ["lightgbm", "hist_gradient_boosting"],
                "ml_model_weights": [1.0, 1.0],
                "rd_checkpoint": str(DEFAULT_RD_CHECKPOINT),
                "partition": "val", "batch_size_tr": 8, "batch_size_rd": 128,
                "tr_epochs": 0, "tr_learning_rate": 0.0002, "tr_weight_decay": 0.0001,
                "tr_patience": 0, "drone_sample_boost": 1.0, "bird_sample_boost": 1.0,
                "balloon_sample_boost": 1.0, "clutter_sample_boost": 2.0,
                "unknown_sample_boost": 1.0, "sampling_mode": "b01_balanced",
                "sampling_protocol": "coverage_plus_boost",
                "manual_sampling_boosts_enabled": False, "augmentation_frame_drop_fraction": 0.10,
                "sampling_class_enabled": [True, True, True, True, True],
                "sampling_mode": "b01_balanced",
                "augmentation_amplitude_scale_max_delta": 0.05, "augmentation_snr_offset_db": 1.0,
                "checkpoint_selection_metric": "macro_f1", "class_weight_mode": "inverse_sqrt",
                "class_balanced_beta": 0.999, "class_weight_floor": 0.25, "class_weight_cap": 2.0,
                "manual_class_loss_weights": None, "manual_class_loss_weights_enabled": False,
                "class_loss_enabled": [True, True, True, True, True],
                "use_cosface": True,
                "lr_scheduler": "none",
                "fusion_mode": "fixed", "fixed_rd_weight": 0.2,
                "fusion_checkpoint": "",
                "folds": 5, "inner_val_fraction": 0.15, "rd_epochs": 0,
                "gate_epochs": 200, "gate_learning_rate": 0.01,
                "gate_weight_decay": 0.0001, "gate_hidden_dim": 32,
                "initial_rd_weight": 0.2,
                **PARTITION_AUGMENTATION_DEFAULTS,
            })
            return
        if request_path == "/api/tr-class-loss-preview":
            query = parse_qs(parsed.query)
            try:
                split = Path(query.get("grouped_split", [str(DEFAULT_GROUPED_SPLIT)])[0]).expanduser().resolve()
                track_index = Path(query.get("track_index", [str(DEFAULT_TRACK_INDEX)])[0]).expanduser().resolve()
                if not split.is_file() or not track_index.is_file():
                    raise ValueError("Fixed split or track index file does not exist")
                raw_targets = query.get("targets", ["489,489,489,489,489"])[0]
                targets = [int(value.strip()) for value in raw_targets.split(",")]
                preview = automatic_tr_class_loss_preview(
                    split, track_index, targets,
                    mode=str(query.get("mode", ["inverse_sqrt"])[0]),
                    beta=float(query.get("beta", ["0.999"])[0]),
                    floor=float(query.get("floor", ["0.5"])[0]),
                    cap=float(query.get("cap", ["3"])[0]),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, preview)
            return
        if request_path == "/api/tr-sampling-preview":
            query = parse_qs(parsed.query)
            try:
                split = Path(query.get("grouped_split", [str(DEFAULT_GROUPED_SPLIT)])[0]).expanduser().resolve()
                track_index = Path(query.get("track_index", [str(DEFAULT_TRACK_INDEX)])[0]).expanduser().resolve()
                if not split.is_file() or not track_index.is_file():
                    raise ValueError("Fixed split or track index file does not exist")
                raw_targets = query.get("targets", ["489,245,245,35,70"])[0]
                targets = [int(value.strip()) for value in raw_targets.split(",")]
                preview = automatic_tr_sampling_preview(
                    split, track_index, targets,
                    mode=str(query.get("mode", ["b01_balanced"])[0]),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, preview)
            return
        if request_path == "/api/tr-balance-preview":
            query = parse_qs(parsed.query)
            try:
                split = Path(query.get("grouped_split", [str(DEFAULT_GROUPED_SPLIT)])[0]).expanduser().resolve()
                track_index = Path(query.get("track_index", [str(DEFAULT_TRACK_INDEX)])[0]).expanduser().resolve()
                if not split.is_file() or not track_index.is_file():
                    raise ValueError("Fixed split or track index file does not exist")
                raw_targets = query.get("targets", ["489,245,245,35,70"])[0]
                targets = [int(value.strip()) for value in raw_targets.split(",")]
                raw_enabled = query.get("enabled", ["1,1,1,1,1"])[0]
                enabled = [bool(int(value.strip())) for value in raw_enabled.split(",")]
                raw_sampling_enabled = query.get("sampling_enabled", ["1,1,1,1,1"])[0]
                sampling_enabled = [bool(int(value.strip())) for value in raw_sampling_enabled.split(",")]
                raw_loss_enabled = query.get("loss_enabled", ["1,1,1,1,1"])[0]
                loss_enabled = [bool(int(value.strip())) for value in raw_loss_enabled.split(",")]
                raw_sampling = query.get("sampling_boosts", [""])[0].strip()
                sampling_boosts = [float(value.strip()) for value in raw_sampling.split(",")] if raw_sampling else None
                raw_manual_loss = query.get("manual_loss_weights", [""])[0].strip()
                manual_loss = [float(value.strip()) for value in raw_manual_loss.split(",")] if raw_manual_loss else None
                preview = automatic_tr_balance_preview(
                    split, track_index, targets, enabled=enabled,
                    sampling_enabled=sampling_enabled, loss_enabled=loss_enabled,
                    sampling_mode=str(query.get("sampling_mode", ["b01_balanced"])[0]),
                    sampling_boosts=sampling_boosts,
                    class_weight_mode=str(query.get("class_weight_mode", ["inverse_sqrt"])[0]),
                    beta=float(query.get("beta", ["0.999"])[0]),
                    floor=float(query.get("floor", ["0.25"])[0]),
                    cap=float(query.get("cap", ["2"])[0]),
                    manual_loss_weights=manual_loss,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, preview)
            return
        if request_path == "/api/checkpoints":
            self.send_json(200, {"items": checkpoint_catalog(), "generated_at": datetime.now().isoformat(timespec="seconds")})
            return
        if request_path == "/api/experiments":
            self.send_json(200, {
                "api_version": 2,
                "items": discover(),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            })
            return
        decisions_match = re.fullmatch(r"/api/experiments/([^/]+)/decisions", request_path)
        if decisions_match:
            name = unquote(decisions_match.group(1))
            if not EXPERIMENT_NAME_RE.fullmatch(name):
                self.send_json(400, {"error": "Invalid experiment name"})
                return
            artifacts = (ROOT / "artifacts").resolve()
            output = (artifacts / name).resolve()
            if output.parent != artifacts or not output.is_dir():
                self.send_json(404, {"error": "Experiment not found"})
                return
            query = parse_qs(parsed.query)
            try:
                branch = str(query.get("branch", ["rd"])[0])
                source = str(query.get("source", ["original"])[0])
                errors_only = str(query.get("errors_only", ["0"])[0]).lower() in {"1", "true", "yes"}
                offset = max(0, int(query.get("offset", [0])[0]))
                limit = min(500, max(1, int(query.get("limit", [100])[0])))
                payload = read_trajectory_decisions(
                    output, branch, errors_only=errors_only,
                    offset=offset, limit=limit, source=source,
                )
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, payload)
            return
        experiment_match = re.fullmatch(r"/api/experiments/([^/]+)", request_path)
        if experiment_match:
            name = unquote(experiment_match.group(1))
            item = next((value for value in discover() if value.get("name") == name), None)
            if item is None:
                self.send_json(404, {"error": "Experiment not found"})
            else:
                self.send_json(200, {"api_version": 2, "item": item})
            return
        if request_path == "/api/trainings":
            body = json.dumps(discover(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if request_path in ("/", "/index.html"):
            body = HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/api/scheduler/settings":
            self.update_scheduler_settings()
            return
        move_match = re.fullmatch(r"/api/scheduler/jobs/([^/]+)/move", self.path)
        if move_match:
            self.move_scheduler_job(unquote(move_match.group(1)))
            return
        confirm_match = re.fullmatch(r"/api/scheduler/jobs/([^/]+)/confirm", self.path)
        if confirm_match:
            self.confirm_scheduler_job(unquote(confirm_match.group(1)))
            return
        cancel_match = re.fullmatch(r"/api/scheduler/jobs/([^/]+)/cancel", self.path)
        if cancel_match:
            self.cancel_scheduler_job(unquote(cancel_match.group(1)))
            return
        if self.path == "/api/trainings":
            self.create_experiment()
            return
        restart_match = re.fullmatch(r"/api/trainings/([^/]+)/restart", self.path)
        if restart_match:
            self.restart_experiment(unquote(restart_match.group(1)))
            return
        supplement_match = re.fullmatch(r"/api/experiments/([^/]+)/supplement-augmentation", self.path)
        if supplement_match:
            self.supplement_augmentation(unquote(supplement_match.group(1)))
            return
        resume_match = re.fullmatch(r"/api/trainings/([^/]+)/resume", self.path)
        if resume_match:
            self.resume_experiment(unquote(resume_match.group(1)))
            return
        match = re.fullmatch(r"/api/trainings/([^/]+)/rename", self.path)
        if not match:
            self.send_error(404)
            return
        old_name = unquote(match.group(1))
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            new_name = str(payload.get("name", "")).strip()
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "Invalid JSON body"})
            return
        if not EXPERIMENT_NAME_RE.fullmatch(new_name):
            self.send_json(400, {"error": "Name must be 1-120 characters: letters, numbers, Chinese, _, -, ."})
            return
        artifacts = (ROOT / "artifacts").resolve()
        old_dir = (artifacts / old_name).resolve()
        new_dir = (artifacts / new_name).resolve()
        if old_dir.parent != artifacts or new_dir.parent != artifacts:
            self.send_json(400, {"error": "Invalid experiment path"})
            return
        if old_name == new_name:
            self.send_json(200, {"name": new_name})
            return
        queued = next((item for item in SCHEDULER.experiment_records()
                       if item.get("name") == old_name and item.get("queue_status") in {"queued", CONFIRMATION_STATE}), None)
        if queued:
            if new_dir.exists() or SCHEDULER.has_name(new_name):
                self.send_json(409, {"error": "An experiment with this name already exists"})
                return
            try:
                SCHEDULER.rename(old_name, new_name)
                rename_experiment_index(old_name, new_name)
            except (OSError, ValueError) as exc:
                self.send_json(500, {"error": f"Rename failed: {exc}"})
                return
            invalidate_discover_cache()
            self.send_json(200, {"name": new_name})
            return
        if not old_dir.is_dir() or not (old_dir / "config.json").exists():
            self.send_json(404, {"error": "Experiment not found"})
            return
        if new_dir.exists():
            self.send_json(409, {"error": "An experiment with this name already exists"})
            return
        current = next((item for item in discover() if item["name"] == old_name), None)
        if current and current.get("running"):
            self.send_json(409, {"error": "Cannot rename a running experiment"})
            return
        try:
            old_dir.rename(new_dir)
            config_path = new_dir / "config.json"
            config = read_json(config_path) or {}
            config["output_dir"] = str(new_dir)
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for suffix in (".log", ".err.log"):
                old_log = artifacts / f"{old_name}{suffix}"
                new_log = artifacts / f"{new_name}{suffix}"
                if old_log.exists() and not new_log.exists():
                    old_log.rename(new_log)
            SCHEDULER.rename(old_name, new_name)
            rename_experiment_index(old_name, new_name)
        except OSError as exc:
            if new_dir.exists() and not old_dir.exists():
                try:
                    new_dir.rename(old_dir)
                except OSError:
                    pass
            self.send_json(500, {"error": f"Rename failed: {exc}"})
            return
        invalidate_discover_cache()
        self.send_json(200, {"name": new_name})

    def read_payload(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def update_scheduler_settings(self):
        try:
            payload = self.read_payload()
            settings = SCHEDULER.update_settings(payload)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        self.send_json(200, settings)

    def move_scheduler_job(self, job_id):
        try:
            payload = self.read_payload()
            SCHEDULER.move(job_id, str(payload.get("direction", "")))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        self.send_json(200, SCHEDULER.snapshot())

    def confirm_scheduler_job(self, job_id):
        try:
            job = SCHEDULER.confirm(job_id)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        invalidate_discover_cache()
        self.send_json(200, {"job": job, "scheduler": SCHEDULER.snapshot()})

    def cancel_scheduler_job(self, job_id):
        try:
            SCHEDULER.cancel(job_id)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        invalidate_discover_cache()
        self.send_json(200, SCHEDULER.snapshot())

    def create_experiment(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "Invalid JSON body"})
            return
        name = str(payload.get("name", "")).strip()
        if not EXPERIMENT_NAME_RE.fullmatch(name):
            self.send_json(400, {"error": "Name must be 1-120 characters: letters, numbers, Chinese, _, -, ."})
            return
        experiment_type = str(payload.get("experiment_type") or "rd_only").strip().lower()
        if experiment_type not in {"rd_only", "tr_only", "tr_checkpoint_eval", "rd_checkpoint_eval", "tr_rd_soft_cascade", "fusion_gate_training", "fusion_gate_calibration"}:
            self.send_json(400, {"error": "Unsupported experiment type"})
            return
        if experiment_type == "tr_only":
            self.create_tr_training(name, payload)
            return
        if experiment_type == "fusion_gate_training":
            self.create_gate_training(name, payload)
            return
        if experiment_type == "fusion_gate_calibration":
            self.create_calibration_gate_training(name, payload)
            return
        if experiment_type != "rd_only":
            self.create_branch_experiment(name, experiment_type, payload)
            return
        try:
            params = {
                "experiment_type": "rd_only",
                "epochs": int(payload.get("epochs", 50)),
                "batch_size": int(payload.get("batch_size", 128)),
                "workers": int(payload.get("workers", 4)),
                "max_train_frames_per_trajectory": int(payload.get("max_train_frames_per_trajectory", 32)),
                "norm_samples": int(payload.get("norm_samples", 2048)),
                "learning_rate": float(payload.get("learning_rate", 0.0003)),
                "weight_decay": float(payload.get("weight_decay", 0.0001)),
                "patience": int(payload.get("patience", 10)),
                "seed": int(payload.get("seed", 42)),
                "velocity_min": float(payload.get("velocity_min", -90)),
                "velocity_max": float(payload.get("velocity_max", 89)),
                "target_width": int(payload.get("target_width", 900)),
                "resampling": str(payload.get("resampling", "db_linear")),
                "normalization": str(payload.get("normalization", "global_z")),
                "input_mode": str(payload.get("input_mode", "rd")),
                "model_head": str(payload.get("model_head", "global")),
                "augmentation": str(payload.get("augmentation", "off")),
                "split_mode": str(payload.get("split_mode", "fixed_grouped")),
                "train_registry": str(payload.get("train_registry", "")).strip(),
                "grouped_split": str(payload.get("grouped_split", "")).strip(),
                "rd_cache": str(payload.get("rd_cache", "")).strip(),
                "skip_test": bool(payload.get("skip_test", False)),
            } | self.partition_augmentation_params(payload)
        except (TypeError, ValueError):
            self.send_json(400, {"error": "Training parameters must be numeric"})
            return
        limits = {
            "epochs": (1, 10000), "batch_size": (1, 65536), "workers": (0, 64),
            "max_train_frames_per_trajectory": (1, 100000), "norm_samples": (1, 1000000),
            "learning_rate": (1e-9, 10.0), "weight_decay": (0.0, 10.0),
            "patience": (1, 10000), "seed": (0, 2**31 - 1),
            "velocity_min": (-10000.0, 10000.0), "velocity_max": (-10000.0, 10000.0),
            "target_width": (8, 10000),
        }
        invalid = [key for key, value in params.items() if key in limits and not limits[key][0] <= value <= limits[key][1]]
        if params["velocity_min"] >= params["velocity_max"]:
            invalid.append("velocity range")
        choices = {
            "resampling": {"db_linear", "power_linear", "area"},
            "normalization": {"global_z", "frame_z", "frame_robust", "minmax", "clip"},
            "input_mode": {"rd", "rd_mask", "rd_peak", "rd_background", "rd_contrast"},
            "model_head": {"global", "spatial_2x8"},
            "augmentation": {"off", "minority_rd"},
            "split_mode": {"fixed_grouped", "random_stratified", "registry_train"},
        }
        invalid.extend(key for key, values in choices.items() if params[key] not in values)
        if params["split_mode"] == "fixed_grouped":
            grouped_split = Path(params["grouped_split"] or DEFAULT_GROUPED_SPLIT).expanduser()
            if not grouped_split.is_file():
                invalid.append("grouped_split")
            else:
                params["grouped_split"] = str(grouped_split.resolve())
            params["train_registry"] = ""
        elif params["split_mode"] == "registry_train":
            registry = Path(params["train_registry"] or DEFAULT_TRAIN_REGISTRY).expanduser()
            if not registry.is_file():
                invalid.append("train_registry")
            else:
                params["train_registry"] = str(registry.resolve())
        else:
            params["train_registry"] = ""
            params["grouped_split"] = ""
        # Cache selection is derived from the final active protocol, rather
        # than inherited from a cloned experiment or a browser's local state.
        # No matching prebuilt cache means on-demand preprocessing, not a
        # submission error and not a stale path.
        params["rd_cache"] = recommended_rd_cache(params)
        if invalid:
            self.send_json(400, {"error": f"Invalid parameter range: {', '.join(invalid)}"})
            return
        dataset_root = Path(str(payload.get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
        if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
            self.send_json(400, {"error": "Dataset root must contain a MAT directory"})
            return
        artifacts = (ROOT / "artifacts").resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        output = (artifacts / name).resolve()
        if output.parent != artifacts:
            self.send_json(400, {"error": "Invalid experiment path"})
            return
        if output.exists() or (artifacts / f"{name}.log").exists() or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name):
            self.send_json(409, {"error": "An experiment or log with this name already exists"})
            return
        try:
            job = SCHEDULER.enqueue(name, dataset_root, params)
        except (OSError, ValueError) as exc:
            self.send_json(500, {"error": f"Failed to queue training: {exc}"})
            return
        invalidate_discover_cache()
        self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE})

    def create_tr_training(self, name, payload):
        """Validate and queue a real B01-compatible TR training run."""
        def existing_file(field, default):
            path = Path(str(payload.get(field) or default)).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"{field} file does not exist")
            return str(path)

        def optional_float(field, default):
            """Use the protocol default for a disabled/omitted form control."""
            value = payload.get(field)
            return float(default if value is None or value == "" else value)

        def optional_bool_list(field, default):
            raw = payload.get(field, default)
            if isinstance(raw, str):
                raw = [item.strip() for item in raw.split(",")]
            if not isinstance(raw, list) or len(raw) != 5:
                raise ValueError(f"{field} requires five values")
            return [
                (str(value).strip().lower() not in {"0", "false", "off", "no", ""}
                 if isinstance(value, str) else bool(value))
                for value in raw
            ]

        def optional_bool(field, default):
            """Parse form booleans without turning the string 'false' into True."""
            value = payload.get(field, default)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "on", "yes"}:
                    return True
                if normalized in {"0", "false", "off", "no", ""}:
                    return False
                raise ValueError(f"{field} must be a boolean")
            return bool(value)

        try:
            partition_augmentation = self.partition_augmentation_params(payload)
            tr_implementation = str(payload.get("tr_implementation", "dl") or "dl").strip().lower()
            if tr_implementation not in {"dl", "ml"}:
                raise ValueError("tr_implementation must be dl or ml")
            if tr_implementation == "ml":
                raw_models = payload.get("ml_models", ["lightgbm", "hist_gradient_boosting"])
                if isinstance(raw_models, str):
                    raw_models = [item.strip() for item in raw_models.split(",") if item.strip()]
                if not isinstance(raw_models, list) or not raw_models:
                    raise ValueError("ML 至少选择一个模型")
                allowed_models = {"lightgbm", "hist_gradient_boosting", "extra_trees", "rbf_svm"}
                if any(str(item) not in allowed_models for item in raw_models):
                    raise ValueError("包含不支持的 TR ML 模型")
                raw_weights = payload.get("ml_model_weights")
                if isinstance(raw_weights, str):
                    raw_weights = [item.strip() for item in raw_weights.split(",") if item.strip()]
                weights = [float(value) for value in (raw_weights or [1.0] * len(raw_models))]
                if len(weights) != len(raw_models) or any(value <= 0 for value in weights):
                    raise ValueError("ML 模型权重必须与模型数量一致且为正数")
                ml_soft_voting = optional_bool("ml_soft_voting", False)
                if ml_soft_voting and len(raw_models) < 2:
                    raise ValueError("软投票至少需要两个成员模型")
                params = {
                    "experiment_type": "tr_only", "tr_implementation": "ml",
                    "grouped_split": existing_file("grouped_split", DEFAULT_GROUPED_SPLIT),
                    "track_index": existing_file("track_index", DEFAULT_TRACK_INDEX),
                    "seed": int(payload.get("seed", 42)),
                    "ml_model_plan": str(payload.get("ml_model_plan", "custom") or "custom"),
                    "ml_models": [str(item) for item in raw_models],
                    "ml_model_weights": weights, "ml_soft_voting": ml_soft_voting,
                    "skip_test": optional_bool("skip_test", True),
                    "partition_augmentation_diagnostics": False,
                    "partition_augmentation_method": "off",
                    "partition_augmentation_train_enabled": [False] * 5,
                    "partition_augmentation_targets_train": [489] * 5,
                    "partition_augmentation_targets_val": [105] * 5,
                    "partition_augmentation_targets_test": [105] * 5,
                }
                artifacts = (ROOT / "artifacts").resolve(); artifacts.mkdir(parents=True, exist_ok=True)
                output = (artifacts / name).resolve()
                if output.parent != artifacts:
                    raise ValueError("Invalid experiment path")
                if (output.exists() or (artifacts / f"{name}.log").exists()
                        or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name)):
                    self.send_json(409, {"error": "An experiment or log with this name already exists"}); return
                try:
                    job = SCHEDULER.enqueue(name, ROOT, params)
                except (OSError, ValueError) as exc:
                    self.send_json(500, {"error": f"Failed to queue TR ML training: {exc}"}); return
                invalidate_discover_cache()
                self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE, "experiment_type": "tr_only"})
                return
            epochs = int(payload.get("epochs", 24))
            batch_size = int(payload.get("batch_size_tr", payload.get("batch_size", 8)))
            workers = int(payload.get("workers", 0))
            learning_rate = optional_float("learning_rate", 0.0002)
            weight_decay = optional_float("weight_decay", 0.0001)
            patience = int(payload.get("patience", 0))
            dropout = optional_float("dropout", 0.1)
            seed = int(payload.get("seed", 42))
            drone_boost = optional_float("drone_sample_boost", 1.0)
            bird_boost = optional_float("bird_sample_boost", 1.0)
            balloon_boost = optional_float("balloon_sample_boost", 1.0)
            clutter_boost = optional_float("clutter_sample_boost", 2.0)
            unknown_boost = optional_float("unknown_sample_boost", 1.0)
            sampling_class_enabled = optional_bool_list("sampling_class_enabled", [True] * 5)
            loss_class_enabled = optional_bool_list("class_loss_enabled", [True] * 5)
            sampling_mode = str(payload.get("sampling_mode", "b01_balanced"))
            sampling_protocol = str(payload.get("sampling_protocol", "coverage_plus_boost"))
            raw_manual_sampling = payload.get("manual_sampling_boosts_enabled", False)
            manual_sampling_enabled = (str(raw_manual_sampling).strip().lower() not in {"0", "false", "off", "no", ""}
                                       if isinstance(raw_manual_sampling, str) else bool(raw_manual_sampling))
            manual_sampling_boosts = payload.get("manual_sampling_boosts")
            if manual_sampling_enabled and manual_sampling_boosts is None:
                manual_sampling_boosts = [payload.get(key) for key in (
                    "drone_sample_boost", "bird_sample_boost", "balloon_sample_boost",
                    "clutter_sample_boost", "unknown_sample_boost")]
            if manual_sampling_boosts is not None:
                if not isinstance(manual_sampling_boosts, list) or len(manual_sampling_boosts) != 5:
                    raise ValueError("Manual sampling boosts must contain five values")
                manual_sampling_boosts = [float(value) for value in manual_sampling_boosts]
            if not manual_sampling_enabled:
                manual_sampling_boosts = None
                drone_boost, bird_boost, balloon_boost, clutter_boost, unknown_boost = (
                    (1.0, 1.0, 1.0, 1.0, 1.0) if sampling_mode == "inverse_frequency"
                    else (1.0, 1.0, 1.0, 2.0, 1.0)
                )
            elif manual_sampling_boosts is not None:
                drone_boost, bird_boost, balloon_boost, clutter_boost, unknown_boost = manual_sampling_boosts
            augmentation_frame_drop_fraction = optional_float("augmentation_frame_drop_fraction", 0.10)
            augmentation_amplitude_scale_max_delta = optional_float("augmentation_amplitude_scale_max_delta", 0.05)
            augmentation_snr_offset_db = optional_float("augmentation_snr_offset_db", 1.0)
            checkpoint_selection_metric = str(payload.get("checkpoint_selection_metric", "macro_f1"))
            class_weight_mode = str(payload.get("class_weight_mode", "inverse_sqrt"))
            class_balanced_beta = optional_float("class_balanced_beta", 0.999)
            class_weight_cap = optional_float("class_weight_cap", 2.0)
            class_weight_floor = optional_float("class_weight_floor", 0.25)
            manual_class_loss_weights = payload.get("manual_class_loss_weights")
            if manual_class_loss_weights is not None:
                if not isinstance(manual_class_loss_weights, list) or len(manual_class_loss_weights) != 5:
                    raise ValueError("Manual class loss weights must contain five values")
                manual_class_loss_weights = [float(value) for value in manual_class_loss_weights]
            lr_scheduler = str(payload.get("lr_scheduler", "none"))
            use_cosface = optional_bool("use_cosface", True)
            if not 1 <= epochs <= 1000 or not 1 <= batch_size <= 4096 or not 0 <= workers <= 64:
                raise ValueError("Invalid TR training epochs, batch size, or worker count")
            if not 0.0 < learning_rate or weight_decay < 0.0 or not 0 <= patience <= epochs or not 0.0 <= dropout < 1.0:
                raise ValueError("Invalid TR training optimizer parameters")
            if sampling_mode not in {"b01_balanced", "inverse_frequency"}:
                raise ValueError("Invalid TR sampling mode")
            if sampling_protocol not in {"coverage_plus_boost", "strict_b01_replacement"}:
                raise ValueError("Invalid TR sampling protocol")
            if not all(0.0 < value <= 20.0 for value in (drone_boost, bird_boost, balloon_boost, clutter_boost, unknown_boost)):
                raise ValueError("Invalid TR sample boost")
            if not 0.0 <= augmentation_frame_drop_fraction <= 0.40 or augmentation_amplitude_scale_max_delta < 0.0 or augmentation_snr_offset_db < 0.0:
                raise ValueError("Invalid trajectory augmentation magnitude")
            if checkpoint_selection_metric not in {"macro_f1", "bird_f1"}:
                raise ValueError("Invalid TR checkpoint selection metric")
            if class_weight_mode not in {"inverse_sqrt", "class_balanced"} or not 0.0 <= class_balanced_beta < 1.0 or class_weight_floor < 0.0 or class_weight_cap < class_weight_floor or lr_scheduler not in {"cosine", "none"}:
                raise ValueError("Invalid TR class-loss weighting parameters")
            if manual_class_loss_weights is not None and not all(0.0 < value <= 20.0 for value in manual_class_loss_weights):
                raise ValueError("Manual class loss weights must be in (0, 20]")
            params = {
                "experiment_type": "tr_only",
                "tr_implementation": "dl",
                "grouped_split": existing_file("grouped_split", DEFAULT_GROUPED_SPLIT),
                "track_index": existing_file("track_index", DEFAULT_TRACK_INDEX),
                "epochs": epochs, "batch_size_tr": batch_size, "workers": workers,
                "learning_rate": learning_rate, "weight_decay": weight_decay,
                "patience": patience, "dropout": dropout, "seed": seed,
                "drone_sample_boost": drone_boost,
                "bird_sample_boost": bird_boost,
                "balloon_sample_boost": balloon_boost,
                "clutter_sample_boost": clutter_boost,
                "unknown_sample_boost": unknown_boost,
                "sampling_mode": sampling_mode,
                "sampling_protocol": sampling_protocol,
                "manual_sampling_boosts_enabled": manual_sampling_enabled,
                "manual_sampling_boosts": manual_sampling_boosts,
                "sampling_class_enabled": sampling_class_enabled,
                "augmentation_frame_drop_fraction": augmentation_frame_drop_fraction,
                "augmentation_amplitude_scale_max_delta": augmentation_amplitude_scale_max_delta,
                "augmentation_snr_offset_db": augmentation_snr_offset_db,
                "checkpoint_selection_metric": checkpoint_selection_metric,
                "class_weight_mode": class_weight_mode,
                "class_balanced_beta": class_balanced_beta,
                "class_weight_floor": class_weight_floor,
                "class_weight_cap": class_weight_cap,
                "manual_class_loss_weights": manual_class_loss_weights,
                "class_loss_enabled": loss_class_enabled,
                "use_cosface": use_cosface,
                "manual_training_copy_targets_enabled": bool(payload.get("manual_training_copy_targets_enabled", False)),
                "lr_scheduler": lr_scheduler,
                "skip_test": bool(payload.get("skip_test", True)),
            } | partition_augmentation
        except (TypeError, ValueError, OSError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        artifacts = (ROOT / "artifacts").resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        output = (artifacts / name).resolve()
        if output.parent != artifacts:
            self.send_json(400, {"error": "Invalid experiment path"})
            return
        if (output.exists() or (artifacts / f"{name}.log").exists()
                or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name)):
            self.send_json(409, {"error": "An experiment or log with this name already exists"})
            return
        try:
            job = SCHEDULER.enqueue(name, ROOT, params)
        except (OSError, ValueError) as exc:
            self.send_json(500, {"error": f"Failed to queue TR training: {exc}"})
            return
        invalidate_discover_cache()
        self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE, "experiment_type": "tr_only"})

    @staticmethod
    def partition_augmentation_params(payload):
        raw_enabled = payload.get("partition_augmentation_diagnostics", True)
        # JSON callers normally send a boolean, but treating the string
        # "false" as truthy made API-created jobs disagree with the checkbox.
        enabled = (str(raw_enabled).strip().lower() not in {"0", "false", "off", "no", ""}
                   if isinstance(raw_enabled, str) else bool(raw_enabled))
        raw_train_enabled = payload.get("partition_augmentation_train_enabled", [True] * 5)
        if isinstance(raw_train_enabled, str):
            raw_train_enabled = [item.strip() for item in raw_train_enabled.split(",")]
        if not isinstance(raw_train_enabled, list) or len(raw_train_enabled) != 5:
            raise ValueError("partition augmentation train switches require five values")
        train_enabled = [
            (str(value).strip().lower() not in {"0", "false", "off", "no", ""}
             if isinstance(value, str) else bool(value))
            for value in raw_train_enabled
        ]
        method = str(payload.get("partition_augmentation_method", "perturbation")).strip().lower()
        if method not in {"perturbation", "smote"}:
            raise ValueError("partition augmentation method must be perturbation or smote")
        params = {"partition_augmentation_diagnostics": enabled,
                  "partition_augmentation_method": method,
                  "partition_augmentation_train_enabled": train_enabled}
        for partition, default in (("train", [489] * 5), ("val", [105] * 5), ("test", [105] * 5)):
            raw = payload.get(f"partition_augmentation_targets_{partition}", default)
            if isinstance(raw, str):
                raw = [item.strip() for item in raw.split(",")]
            if not isinstance(raw, list) or len(raw) != 5:
                raise ValueError(f"partition augmentation {partition} targets require five values")
            values = [int(value) for value in raw]
            if any(value < 0 or value > 10000 for value in values):
                raise ValueError(f"partition augmentation {partition} targets must be 0–10000")
            params[f"partition_augmentation_targets_{partition}"] = values
        return params

    def create_gate_training(self, name, payload):
        """Validate and queue strict train-partition OOF gate fitting."""
        def existing_file(field, default):
            path = Path(str(payload.get(field) or default)).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"{field} file does not exist")
            return str(path)

        try:
            dataset_root = Path(str(payload.get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
            if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
                raise ValueError("Dataset root must contain a MAT directory")
            folds = int(payload.get("folds", 5))
            inner_fraction = float(payload.get("inner_val_fraction", 0.15))
            tr_epochs = int(payload.get("tr_epochs", 0))
            rd_epochs = int(payload.get("rd_epochs", 0))
            batch_size_tr = int(payload.get("batch_size_tr", 8))
            batch_size_rd = int(payload.get("batch_size_rd", 128))
            workers = int(payload.get("workers", 0))
            gate_epochs = int(payload.get("gate_epochs", 200))
            gate_lr = float(payload.get("gate_learning_rate", 0.01))
            gate_decay = float(payload.get("gate_weight_decay", 0.0001))
            hidden_dim = int(payload.get("gate_hidden_dim", 32))
            initial_weight = float(payload.get("initial_rd_weight", 0.2))
            seed = int(payload.get("seed", 42))
            if not 2 <= folds <= 10:
                raise ValueError("OOF folds must be between 2 and 10")
            if not 0.05 <= inner_fraction <= 0.40:
                raise ValueError("Inner validation fraction must be between 0.05 and 0.40")
            if not 0 <= tr_epochs <= 1000 or not 0 <= rd_epochs <= 1000:
                raise ValueError("OOF branch epochs must be between 0 and 1000")
            if not 1 <= batch_size_tr <= 4096 or not 1 <= batch_size_rd <= 65536:
                raise ValueError("Invalid OOF batch size")
            if not 0 <= workers <= 64 or gate_epochs < 1 or hidden_dim < 1:
                raise ValueError("Invalid OOF worker, gate epoch, or hidden size")
            if gate_lr <= 0 or gate_decay < 0 or not 0 < initial_weight < 1:
                raise ValueError("Invalid gate optimizer or initial RD weight")
            params = {
                "experiment_type": "fusion_gate_training",
                "grouped_split": existing_file("grouped_split", DEFAULT_GROUPED_SPLIT),
                "track_index": existing_file("track_index", DEFAULT_TRACK_INDEX),
                "tr_checkpoint": existing_file("tr_checkpoint", DEFAULT_TR_CHECKPOINT),
                "rd_checkpoint": existing_file("rd_checkpoint", DEFAULT_RD_CHECKPOINT),
                "folds": folds, "inner_val_fraction": inner_fraction,
                "tr_epochs": tr_epochs, "rd_epochs": rd_epochs,
                "batch_size_tr": batch_size_tr, "batch_size_rd": batch_size_rd,
                "workers": workers, "gate_epochs": gate_epochs,
                "gate_learning_rate": gate_lr, "gate_weight_decay": gate_decay,
                "gate_hidden_dim": hidden_dim, "initial_rd_weight": initial_weight,
                "seed": seed,
            }
            rd_cache = str(payload.get("rd_cache") or "").strip()
            if rd_cache:
                cache_path = Path(rd_cache).expanduser().resolve()
                if not cache_path.is_dir():
                    raise ValueError("rd_cache directory does not exist")
                params["rd_cache"] = str(cache_path)
        except (TypeError, ValueError, OSError) as exc:
            self.send_json(400, {"error": str(exc)})
            return

        artifacts = (ROOT / "artifacts").resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        output = (artifacts / name).resolve()
        if output.parent != artifacts:
            self.send_json(400, {"error": "Invalid experiment path"})
            return
        if (output.exists() or (artifacts / f"{name}.log").exists()
                or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name)):
            self.send_json(409, {"error": "An experiment or log with this name already exists"})
            return
        try:
            job = SCHEDULER.enqueue(name, dataset_root, params)
        except (OSError, ValueError) as exc:
            self.send_json(500, {"error": f"Failed to queue OOF gate training: {exc}"})
            return
        invalidate_discover_cache()
        self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE,
                             "experiment_type": "fusion_gate_training"})

    def create_calibration_gate_training(self, name, payload):
        """Queue fast gate fitting from frozen TR/RD checkpoint predictions."""
        def existing_file(field, default):
            path = Path(str(payload.get(field) or default)).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"{field} file does not exist")
            return str(path)

        try:
            dataset_root = Path(str(payload.get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
            if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
                raise ValueError("Dataset root must contain a MAT directory")
            batch_size_tr = int(payload.get("batch_size_tr", 8))
            batch_size_rd = int(payload.get("batch_size_rd", 128))
            workers = int(payload.get("workers", 0))
            gate_epochs = int(payload.get("gate_epochs", 200))
            gate_lr = float(payload.get("gate_learning_rate", 0.01))
            gate_decay = float(payload.get("gate_weight_decay", 0.0001))
            hidden_dim = int(payload.get("gate_hidden_dim", 32))
            initial_weight = float(payload.get("initial_rd_weight", 0.2))
            # An older monitor hid this field for calibration and serialized
            # it as JSON null.  Keep the default reproducible in that case.
            seed = int(payload.get("seed") or 42)
            if not 1 <= batch_size_tr <= 4096 or not 1 <= batch_size_rd <= 65536 or not 0 <= workers <= 64:
                raise ValueError("Invalid calibration inference batch size or worker count")
            if gate_epochs < 1 or hidden_dim < 1 or gate_lr <= 0 or gate_decay < 0 or not 0 < initial_weight < 1:
                raise ValueError("Invalid calibration gate parameters")
            params = {
                "experiment_type": "fusion_gate_calibration",
                "grouped_split": existing_file("grouped_split", DEFAULT_GROUPED_SPLIT),
                "track_index": existing_file("track_index", DEFAULT_TRACK_INDEX),
                "tr_checkpoint": existing_file("tr_checkpoint", DEFAULT_TR_CHECKPOINT),
                "rd_checkpoint": existing_file("rd_checkpoint", DEFAULT_RD_CHECKPOINT),
                "calibration_partition": "val", "batch_size_tr": batch_size_tr,
                "batch_size_rd": batch_size_rd, "workers": workers,
                "gate_epochs": gate_epochs, "gate_learning_rate": gate_lr,
                "gate_weight_decay": gate_decay, "gate_hidden_dim": hidden_dim,
                "initial_rd_weight": initial_weight, "seed": seed,
            }
            rd_cache = str(payload.get("rd_cache") or "").strip()
            if rd_cache:
                cache_path = Path(rd_cache).expanduser().resolve()
                if not cache_path.is_dir():
                    raise ValueError("rd_cache directory does not exist")
                params["rd_cache"] = str(cache_path)
        except (TypeError, ValueError, OSError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        artifacts = (ROOT / "artifacts").resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        output = (artifacts / name).resolve()
        if output.parent != artifacts:
            self.send_json(400, {"error": "Invalid experiment path"})
            return
        if (output.exists() or (artifacts / f"{name}.log").exists()
                or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name)):
            self.send_json(409, {"error": "An experiment or log with this name already exists"})
            return
        try:
            job = SCHEDULER.enqueue(name, dataset_root, params)
        except (OSError, ValueError) as exc:
            self.send_json(500, {"error": f"Failed to queue calibration gate training: {exc}"})
            return
        invalidate_discover_cache()
        self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE,
                             "experiment_type": "fusion_gate_calibration"})

    def create_branch_experiment(self, name, experiment_type, payload):
        """Validate and queue checkpoint evaluation or TR-RD soft-cascade evaluation."""
        def existing_file(field, default):
            path = Path(str(payload.get(field) or default)).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"{field} file does not exist")
            return str(path)

        try:
            partition_augmentation = self.partition_augmentation_params(payload)
            partition = str(payload.get("partition") or "val").lower()
            allowed_partitions = {"train", "val", "test"}
            if partition not in allowed_partitions:
                raise ValueError("Partition must be train, val, or test")
            workers = int(payload.get("workers", 0))
            if not 0 <= workers <= 64:
                raise ValueError("Invalid worker count")
            dataset_root = Path(str(payload.get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
            if experiment_type == "rd_checkpoint_eval":
                if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
                    raise ValueError("Dataset root must contain a MAT directory")
                batch_size_rd = int(payload.get("batch_size_rd", 128))
                if not 1 <= batch_size_rd <= 65536:
                    raise ValueError("Invalid RD batch-size range")
                params = {
                    "experiment_type": experiment_type,
                    "rd_checkpoint": existing_file("rd_checkpoint", DEFAULT_RD_CHECKPOINT),
                    "partition": partition,
                    "batch_size_rd": batch_size_rd,
                    "workers": workers,
                } | partition_augmentation
                rd_cache = str(payload.get("rd_cache") or "").strip()
                if rd_cache:
                    cache_path = Path(rd_cache).expanduser().resolve()
                    if not cache_path.is_dir():
                        raise ValueError("rd_cache directory does not exist")
                    params["rd_cache"] = str(cache_path)
            else:
                batch_size_tr = int(payload.get("batch_size_tr", 32))
                if not 1 <= batch_size_tr <= 65536:
                    raise ValueError("Invalid TR batch-size range")
                params = {
                    "experiment_type": experiment_type,
                    "grouped_split": existing_file("grouped_split", DEFAULT_GROUPED_SPLIT),
                    "track_index": existing_file("track_index", DEFAULT_TRACK_INDEX),
                    "tr_checkpoint": existing_file("tr_checkpoint", DEFAULT_TR_CHECKPOINT),
                    "partition": partition,
                    "batch_size_tr": batch_size_tr,
                    "workers": workers,
                } | partition_augmentation
            if experiment_type == "tr_rd_soft_cascade":
                if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
                    raise ValueError("Dataset root must contain a MAT directory")
                batch_size_rd = int(payload.get("batch_size_rd", 128))
                if not 1 <= batch_size_rd <= 65536:
                    raise ValueError("Invalid RD batch-size range")
                fusion_mode = str(payload.get("fusion_mode") or "fixed")
                if fusion_mode not in {"fixed", "quality_classwise"}:
                    raise ValueError("Invalid fusion mode")
                raw_weights = payload.get("fixed_rd_weight", 0.2)
                if fusion_mode == "fixed":
                    if isinstance(raw_weights, str):
                        raw_weights = [value.strip() for value in raw_weights.split(",") if value.strip()]
                    elif not isinstance(raw_weights, list):
                        raw_weights = [raw_weights]
                    weights = [float(value) for value in raw_weights]
                    if len(weights) not in {1, 5} or any(not 0.0 <= value <= 1.0 for value in weights):
                        raise ValueError("RD weight must contain one or five values between 0 and 1")
                else:
                    weights = []
                params.update({
                    "rd_checkpoint": existing_file("rd_checkpoint", DEFAULT_RD_CHECKPOINT),
                    "batch_size_rd": batch_size_rd,
                    "fusion_mode": fusion_mode,
                    "fixed_rd_weight": weights,
                })
                rd_cache = str(payload.get("rd_cache") or "").strip()
                if rd_cache:
                    cache_path = Path(rd_cache).expanduser().resolve()
                    if not cache_path.is_dir():
                        raise ValueError("rd_cache directory does not exist")
                    params["rd_cache"] = str(cache_path)
                fusion_checkpoint = str(payload.get("fusion_checkpoint") or "").strip()
                if fusion_mode == "quality_classwise":
                    if not fusion_checkpoint:
                        raise ValueError("quality_classwise requires fusion_checkpoint")
                    params["fusion_checkpoint"] = existing_file("fusion_checkpoint", fusion_checkpoint)
            elif experiment_type == "tr_checkpoint_eval" and not dataset_root.exists():
                # TR evaluation does not read the MAT dataset; retain a stable
                # scheduler field even when the RD dataset is offline.
                dataset_root = ROOT
        except (TypeError, ValueError, OSError) as exc:
            self.send_json(400, {"error": str(exc)})
            return

        artifacts = (ROOT / "artifacts").resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        output = (artifacts / name).resolve()
        if output.parent != artifacts:
            self.send_json(400, {"error": "Invalid experiment path"})
            return
        if (output.exists() or (artifacts / f"{name}.log").exists()
                or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name)):
            self.send_json(409, {"error": "An experiment or log with this name already exists"})
            return
        try:
            job = SCHEDULER.enqueue(name, dataset_root, params)
        except (OSError, ValueError) as exc:
            self.send_json(500, {"error": f"Failed to queue experiment: {exc}"})
            return
        invalidate_discover_cache()
        self.send_json(201, {
            "name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE,
            "experiment_type": experiment_type,
        })

    def supplement_augmentation(self, name):
        """Run only the missing partition-local virtual-trajectory diagnostic in place."""
        current = next((item for item in discover() if item.get("name") == name), None)
        if not current:
            self.send_json(404, {"error": "Experiment not found"})
            return
        if current.get("status") != "completed" or current.get("running"):
            self.send_json(409, {"error": "Only a completed, idle evaluation can receive an augmentation supplement"})
            return
        config = current.get("config") or {}
        experiment_type = str(current.get("experiment_type") or config.get("experiment_type") or "").lower()
        if experiment_type not in {"rd_checkpoint_eval", "tr_checkpoint_eval", "tr_rd_soft_cascade"}:
            self.send_json(400, {"error": "Only RD, TR, or fusion evaluations support augmentation supplements"})
            return
        output = (ROOT / "artifacts" / name).resolve()
        if output.parent != (ROOT / "artifacts").resolve() or not output.is_dir():
            self.send_json(404, {"error": "Experiment output directory not found"})
            return
        if any((output / filename).is_file() for filename in (
            "test_augmented_trajectory_metrics.json", "validation_augmented_best.json",
            "augmented_metrics.json", "trajectory_decisions_augmented.jsonl",
            "trajectory_decisions_test_augmented.jsonl")):
            self.send_json(409, {"error": "该实验已经有扩增诊断结果，无需重复补充"})
            return
        scheduler_record = next((item for item in SCHEDULER.experiment_records() if item.get("name") == name), None)
        old_params = (scheduler_record or {}).get("params") or {}
        partition = str(config.get("partition") or config.get("calibration_partition") or old_params.get("partition") or "val").lower()
        target_defaults = {"train": [489] * 5, "val": [105] * 5, "test": [105] * 5}
        diag = config.get("partition_augmentation_diagnostics")
        if isinstance(diag, dict):
            section = diag.get(partition) if isinstance(diag.get(partition), dict) else diag.get("partition")
            if isinstance(section, dict) and isinstance(section.get("targets"), list) and len(section["targets"]) == 5:
                target_defaults[partition] = [int(value) for value in section["targets"]]
        trajectory_aug = config.get("trajectory_augmentation") if isinstance(config.get("trajectory_augmentation"), dict) else {}
        if isinstance(trajectory_aug.get("targets"), list) and len(trajectory_aug["targets"]) == 5:
            target_defaults["train"] = [int(value) for value in trajectory_aug["targets"]]

        def cfg_path(*keys):
            for key in keys:
                value = config.get(key) or old_params.get(key)
                if isinstance(value, dict):
                    value = value.get("manifest") or value.get("path") or value.get("checkpoint")
                if value:
                    return str(Path(str(value)).expanduser().resolve())
            return ""

        params = {
            "experiment_type": experiment_type,
            "partition": partition,
            "workers": int(config.get("workers", old_params.get("workers", 0)) or 0),
            "partition_augmentation_diagnostics": True,
            "partition_augmentation_method": str(
                config.get("partition_augmentation_method")
                or (diag.get("method") if isinstance(diag, dict) else "")
                or trajectory_aug.get("method")
                or old_params.get("partition_augmentation_method")
                or "perturbation"
            ),
            "partition_augmentation_train_enabled": old_params.get("partition_augmentation_train_enabled", [True] * 5),
            "partition_augmentation_targets_train": target_defaults["train"],
            "partition_augmentation_targets_val": target_defaults["val"],
            "partition_augmentation_targets_test": target_defaults["test"],
        }
        dataset_root = Path(str(config.get("dataset_root") or (scheduler_record or {}).get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
        if experiment_type == "rd_checkpoint_eval":
            params.update({
                "rd_checkpoint": cfg_path("rd_checkpoint", "checkpoint"),
                "batch_size_rd": int(config.get("batch_size_rd", old_params.get("batch_size_rd", 128)) or 128),
                "rd_cache": str(config.get("rd_cache") or old_params.get("rd_cache") or ""),
            })
            if not params["rd_checkpoint"] or not Path(params["rd_checkpoint"]).is_file():
                self.send_json(400, {"error": "原评估的 RD checkpoint 不存在"})
                return
        elif experiment_type == "tr_checkpoint_eval":
            params.update({
                "grouped_split": cfg_path("grouped_split", "split"),
                "track_index": cfg_path("track_index"),
                "tr_checkpoint": cfg_path("tr_checkpoint", "checkpoint"),
                "batch_size_tr": int(config.get("batch_size", old_params.get("batch_size_tr", 32)) or 32),
            })
            for key, message in (("grouped_split", "原评估的固定划分文件不存在"), ("track_index", "原评估的航迹索引不存在"), ("tr_checkpoint", "原评估的 TR checkpoint 不存在")):
                if not params[key] or not Path(params[key]).is_file():
                    self.send_json(400, {"error": message})
                    return
            dataset_root = ROOT
        else:
            params.update({
                "grouped_split": cfg_path("grouped_split", "split"),
                "track_index": cfg_path("track_index"),
                "tr_checkpoint": cfg_path("tr_checkpoint"),
                "rd_checkpoint": cfg_path("rd_checkpoint"),
                "batch_size_tr": int(config.get("batch_size_tr", old_params.get("batch_size_tr", 32)) or 32),
                "batch_size_rd": int(config.get("batch_size_rd", old_params.get("batch_size_rd", 128)) or 128),
                "fusion_mode": config.get("fusion_mode", old_params.get("fusion_mode", "fixed")),
                "fixed_rd_weight": config.get("fixed_rd_weight", old_params.get("fixed_rd_weight", [0.2])),
                "fusion_checkpoint": cfg_path("fusion_checkpoint") if config.get("fusion_mode") == "quality_classwise" else "",
                "rd_cache": str(config.get("rd_cache") or old_params.get("rd_cache") or ""),
            })
            for key, message in (("grouped_split", "原融合评估的固定划分文件不存在"), ("track_index", "原融合评估的航迹索引不存在"), ("tr_checkpoint", "原融合评估的 TR checkpoint 不存在"), ("rd_checkpoint", "原融合评估的 RD checkpoint 不存在")):
                if not params[key] or not Path(params[key]).is_file():
                    self.send_json(400, {"error": message})
                    return
        if not dataset_root.exists():
            self.send_json(400, {"error": "原评估数据目录不存在"})
            return
        try:
            job = SCHEDULER.supplement_augmentation_in_place(name, dataset_root, params)
        except (OSError, ValueError, TypeError) as exc:
            self.send_json(409, {"error": str(exc)})
            return
        invalidate_discover_cache()
        self.send_json(202, {"name": name, "queue_id": job["id"], "status": job["status"],
                             "experiment_type": experiment_type, "in_place": True,
                             "message": "已在原实验目录排队补充扩增诊断"})

    def restart_experiment(self, old_name):
        """Queue a fresh run from a terminal failed/incomplete experiment."""
        current = next((item for item in discover() if item["name"] == old_name), None)
        if not current:
            self.send_json(404, {"error": "Experiment not found"})
            return
        if current.get("running") or current.get("status") in {CONFIRMATION_STATE, "queued", "starting", "running"}:
            self.send_json(409, {"error": "Only failed or incomplete experiments can be restarted"})
            return
        if current.get("status") not in {"failed", "incomplete"}:
            self.send_json(409, {"error": "Only failed or incomplete experiments can be restarted"})
            return
        config = current.get("config") or {}
        experiment_type = current.get("experiment_type") or config.get("experiment_type") or "rd_only"
        if experiment_type in {"tr_only", "tr_checkpoint_eval", "rd_checkpoint_eval", "tr_rd_soft_cascade", "fusion_gate_training", "fusion_gate_calibration"}:
            provenance = (current.get("fusion") or {}).get("provenance") or {}
            params = dict(current.get("params") or {})
            if experiment_type == "tr_only":
                split_info = config.get("split") if isinstance(config.get("split"), dict) else {}
                params.update({
                    "experiment_type": "tr_only",
                    "grouped_split": config.get("grouped_split") or split_info.get("manifest") or str(DEFAULT_GROUPED_SPLIT),
                    "track_index": config.get("track_index") or str(DEFAULT_TRACK_INDEX),
                    "epochs": config.get("epochs", 24),
                    "batch_size_tr": config.get("batch_size_tr", config.get("batch_size", 8)),
                    "workers": config.get("workers", 0),
                    "learning_rate": config.get("learning_rate", 0.0002),
                    "weight_decay": config.get("weight_decay", 0.0001),
                    "patience": config.get("patience", 0), "seed": config.get("seed", 42),
                    "drone_sample_boost": config.get("drone_sample_boost", 1.0),
                    "bird_sample_boost": config.get("bird_sample_boost", 1.0),
                    "balloon_sample_boost": config.get("balloon_sample_boost", 1.0),
                    "clutter_sample_boost": config.get("clutter_sample_boost", 2.0),
                    "unknown_sample_boost": config.get("unknown_sample_boost", 1.0),
                    "manual_sampling_boosts_enabled": config.get("manual_sampling_boosts_enabled", False),
                    "sampling_mode": config.get("sampling_mode", "b01_balanced"),
                    "sampling_protocol": config.get("sampling_protocol", "coverage_plus_boost"),
                    "sampling_class_enabled": config.get("sampling_class_enabled", [True] * 5),
                    "class_loss_enabled": config.get("class_loss_enabled", [True] * 5),
                    "augmentation_frame_drop_fraction": (config.get("trajectory_augmentation") or {}).get("frame_drop_fraction", config.get("augmentation_frame_drop_fraction", 0.10)),
                    "augmentation_amplitude_scale_max_delta": (config.get("trajectory_augmentation") or {}).get("amplitude_scale_max_delta", config.get("augmentation_amplitude_scale_max_delta", 0.05)),
                    "augmentation_snr_offset_db": (config.get("trajectory_augmentation") or {}).get("snr_offset_db", config.get("augmentation_snr_offset_db", 1.0)),
                    "checkpoint_selection_metric": config.get("checkpoint_selection_metric", "macro_f1"),
                    "class_weight_mode": config.get("class_weight_mode", "inverse_sqrt"),
                    "class_balanced_beta": config.get("class_balanced_beta", 0.999),
                    "class_weight_floor": config.get("class_weight_floor", 0.5),
                    "class_weight_cap": config.get("class_weight_cap", 3.0),
                    "manual_class_loss_weights": config.get("manual_class_loss_weights"),
                    "use_cosface": config.get("use_cosface", True),
                    "lr_scheduler": config.get("lr_scheduler", "cosine"),
                    "skip_test": config.get("test_deferred", True),
                    "partition_augmentation_diagnostics": (config.get("partition_augmentation_diagnostics") or {}).get("enabled", True),
                    "partition_augmentation_method": config.get("partition_augmentation_method") or (config.get("trajectory_augmentation") or {}).get("method", "perturbation"),
                    "partition_augmentation_targets_train": (config.get("trajectory_augmentation") or {}).get("targets", [489] * 5),
                    "partition_augmentation_targets_val": ((config.get("partition_augmentation_diagnostics") or {}).get("val") or {}).get("targets", [105] * 5),
                    "partition_augmentation_targets_test": ((config.get("partition_augmentation_diagnostics") or {}).get("test") or {}).get("targets", [105] * 5),
                })
                dataset_root = ROOT
                artifacts = (ROOT / "artifacts").resolve()
                name = old_name + "_retry"; index = 2
                while ((artifacts / name).exists() or (artifacts / f"{name}.log").exists()
                       or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name)):
                    name = f"{old_name}_retry{index}"; index += 1
                try:
                    job = SCHEDULER.enqueue(name, dataset_root, params)
                except (OSError, ValueError, TypeError) as exc:
                    self.send_json(500, {"error": f"Failed to restart TR training: {exc}"})
                    return
                invalidate_discover_cache()
                self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE, "source": old_name, "experiment_type": "tr_only"})
                return
            if experiment_type == "rd_checkpoint_eval":
                params.update({
                    "experiment_type": "rd_checkpoint_eval",
                    "rd_checkpoint": config.get("rd_checkpoint") or config.get("checkpoint") or str(DEFAULT_RD_CHECKPOINT),
                    "partition": config.get("partition") or "val",
                    "batch_size_rd": config.get("batch_size_rd", config.get("batch_size", 128)),
                    "workers": config.get("workers", 0),
                    "rd_cache": config.get("rd_cache") or "",
                })
                dataset_root = Path(str(config.get("dataset_root") or current.get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
                if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
                    self.send_json(400, {"error": "Dataset root is unavailable for RD checkpoint evaluation restart"})
                    return
                artifacts = (ROOT / "artifacts").resolve()
                name = old_name + "_retry"; index = 2
                while ((artifacts / name).exists() or (artifacts / f"{name}.log").exists()
                       or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name)):
                    name = f"{old_name}_retry{index}"; index += 1
                try:
                    job = SCHEDULER.enqueue(name, dataset_root, params)
                except (OSError, ValueError, TypeError) as exc:
                    self.send_json(500, {"error": f"Failed to restart RD checkpoint evaluation: {exc}"})
                    return
                invalidate_discover_cache()
                self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE, "source": old_name,
                                     "experiment_type": "rd_checkpoint_eval"})
                return
            params.update({
                "experiment_type": experiment_type,
                "grouped_split": config.get("grouped_split") or config.get("split") or provenance.get("grouped_split") or str(DEFAULT_GROUPED_SPLIT),
                "track_index": config.get("track_index") or str(DEFAULT_TRACK_INDEX),
                "tr_checkpoint": config.get("tr_checkpoint") or config.get("checkpoint") or provenance.get("tr_checkpoint") or str(DEFAULT_TR_CHECKPOINT),
                "partition": config.get("partition") or provenance.get("partition") or "val",
                "batch_size_tr": config.get("batch_size_tr") or config.get("batch_size") or 32,
                "workers": config.get("workers", 0),
                "partition_augmentation_diagnostics": (config.get("partition_augmentation_diagnostics") or {}).get("enabled", True),
                "partition_augmentation_method": config.get("partition_augmentation_method") or (config.get("partition_augmentation_diagnostics") or {}).get("method", "perturbation"),
                "partition_augmentation_targets_train": ((config.get("partition_augmentation_diagnostics") or {}).get("train") or {}).get("targets", [489] * 5),
                "partition_augmentation_targets_val": ((config.get("partition_augmentation_diagnostics") or {}).get("val") or {}).get("targets", [105] * 5),
                "partition_augmentation_targets_test": ((config.get("partition_augmentation_diagnostics") or {}).get("test") or {}).get("targets", [105] * 5),
            })
            if experiment_type == "tr_rd_soft_cascade":
                params.update({
                    "rd_checkpoint": config.get("rd_checkpoint") or provenance.get("rd_checkpoint") or str(DEFAULT_RD_CHECKPOINT),
                    "batch_size_rd": config.get("batch_size_rd", 128),
                    "fusion_mode": config.get("fusion_mode") or provenance.get("fusion_mode") or "fixed",
                    "fixed_rd_weight": config.get("fixed_rd_weight") or provenance.get("fixed_rd_weight") or [0.2],
                    "rd_cache": config.get("rd_cache") or provenance.get("rd_cache") or "",
                    "fusion_checkpoint": config.get("fusion_checkpoint") or provenance.get("fusion_checkpoint") or "",
                })
            elif experiment_type in {"fusion_gate_training", "fusion_gate_calibration"}:
                params.update({
                    "rd_checkpoint": config.get("rd_checkpoint") or str(DEFAULT_RD_CHECKPOINT),
                    "batch_size_rd": config.get("batch_size_rd", 128),
                    "gate_epochs": config.get("gate_epochs", 200),
                    "gate_learning_rate": config.get("gate_learning_rate", 0.01),
                    "gate_weight_decay": config.get("gate_weight_decay", 0.0001),
                    "gate_hidden_dim": config.get("gate_hidden_dim", 32),
                    "initial_rd_weight": config.get("initial_rd_weight", 0.2),
                    "seed": config.get("seed", 42),
                    "rd_cache": config.get("rd_cache") or "",
                })
                if experiment_type == "fusion_gate_training":
                    params.update({"folds": config.get("folds", 5),
                                   "inner_val_fraction": config.get("inner_val_fraction", 0.15),
                                   "tr_epochs": config.get("tr_epochs", 0), "rd_epochs": config.get("rd_epochs", 0)})
                else:
                    params["calibration_partition"] = "val"
            dataset_root = Path(str(config.get("dataset_root") or current.get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
            if experiment_type in {"tr_rd_soft_cascade", "fusion_gate_training", "fusion_gate_calibration"} and (not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir()):
                self.send_json(400, {"error": "Dataset root is unavailable for fusion restart"})
                return
            artifacts = (ROOT / "artifacts").resolve()
            name = old_name + "_retry"
            index = 2
            while ((artifacts / name).exists() or (artifacts / f"{name}.log").exists()
                   or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name)):
                name = f"{old_name}_retry{index}"
                index += 1
            try:
                job = SCHEDULER.enqueue(name, dataset_root, params)
            except (OSError, ValueError, TypeError) as exc:
                self.send_json(500, {"error": f"Failed to restart experiment: {exc}"})
                return
            invalidate_discover_cache()
            self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE, "source": old_name,
                                 "experiment_type": experiment_type})
            return
        velocity = config.get("velocity_preprocessing") or {}
        interpolation = config.get("resampling") or velocity.get("interpolation") or "db_linear"
        interpolation = {"linear_in_db": "db_linear", "linear_in_power": "power_linear"}.get(interpolation, interpolation)
        params = {
            "epochs": config.get("epochs", 50), "batch_size": config.get("batch_size", 128),
            "workers": config.get("workers", 4),
            "max_train_frames_per_trajectory": config.get("max_train_frames_per_trajectory", 32),
            "norm_samples": config.get("norm_samples", 2048), "learning_rate": config.get("learning_rate", 0.0003),
            "weight_decay": config.get("weight_decay", 0.0001), "patience": config.get("patience", 10),
            "seed": config.get("seed", 42), "velocity_min": config.get("velocity_min", -90),
            "velocity_max": config.get("velocity_max", 89),
            "target_width": config.get("target_width", velocity.get("target_width", 900)),
            "resampling": interpolation, "normalization": config.get("normalization", "global_z"),
            "input_mode": config.get("input_mode", "rd"), "model_head": config.get("model_head", "global"),
            "augmentation": config.get("augmentation", "off"),
            "split_mode": config.get("split", {}).get("mode", config.get("split_mode", "registry_train")),
            "train_registry": config.get("split", {}).get("registry_path", config.get("train_registry", "")),
            "grouped_split": config.get("split", {}).get("manifest", config.get("grouped_split", "")),
            "skip_test": config.get("skip_test", False),
            "partition_augmentation_diagnostics": (config.get("partition_augmentation_diagnostics") or {}).get("enabled", True),
            "partition_augmentation_targets_train": ((config.get("partition_augmentation_diagnostics") or {}).get("train") or {}).get("targets", [489] * 5),
            "partition_augmentation_targets_val": ((config.get("partition_augmentation_diagnostics") or {}).get("val") or {}).get("targets", [105] * 5),
            "partition_augmentation_targets_test": ((config.get("partition_augmentation_diagnostics") or {}).get("test") or {}).get("targets", [105] * 5),
        }
        dataset_root = Path(str(config.get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
        if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
            dataset_root = DEFAULT_DATASET_ROOT
        if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
            self.send_json(400, {"error": "Dataset root is unavailable"})
            return
        artifacts = (ROOT / "artifacts").resolve()
        suffix = "_retry"
        name = old_name + suffix
        index = 2
        while (artifacts / name).exists() or (artifacts / f"{name}.log").exists() or (artifacts / f"{name}.err.log").exists() or SCHEDULER.has_name(name):
            name = f"{old_name}{suffix}{index}"
            index += 1
        try:
            job = SCHEDULER.enqueue(name, dataset_root, params)
        except (OSError, ValueError, TypeError) as exc:
            self.send_json(500, {"error": f"Failed to restart training: {exc}"})
            return
        invalidate_discover_cache()
        self.send_json(201, {"name": name, "queue_id": job["id"], "status": CONFIRMATION_STATE, "source": old_name})

    def resume_experiment(self, old_name):
        """Continue a terminal experiment from its last completed epoch checkpoint."""
        current = next((item for item in discover() if item["name"] == old_name), None)
        if not current:
            self.send_json(404, {"error": "Experiment not found"})
            return
        if current.get("running") or current.get("status") in {CONFIRMATION_STATE, "queued", "starting", "running"}:
            self.send_json(409, {"error": "Only terminal experiments can be resumed"})
            return
        if current.get("status") not in {"failed", "incomplete", "cancelled"}:
            self.send_json(409, {"error": "Only failed, incomplete, or cancelled experiments can be resumed"})
            return
        output = Path(current["output_dir"])
        checkpoint = output / "last.pt"
        resume_mode = "exact_last_epoch"
        if not checkpoint.is_file():
            checkpoint = output / "best.pt"
            resume_mode = "legacy_best_warm_start"
        if not checkpoint.is_file():
            self.send_json(409, {"error": "No last.pt or best.pt checkpoint is available"})
            return
        config = current.get("config") or {}
        if current.get("experiment_type") == "tr_only":
            split_info = config.get("split") if isinstance(config.get("split"), dict) else {}
            params = {
                "experiment_type": "tr_only",
                "grouped_split": config.get("grouped_split") or split_info.get("manifest") or str(DEFAULT_GROUPED_SPLIT),
                "track_index": config.get("track_index") or str(DEFAULT_TRACK_INDEX),
                "epochs": config.get("epochs", 24),
                "batch_size_tr": config.get("batch_size_tr", config.get("batch_size", 8)),
                "workers": config.get("workers", 0),
                "learning_rate": config.get("learning_rate", 0.0002),
                "weight_decay": config.get("weight_decay", 0.0001),
                "patience": config.get("patience", 0), "dropout": config.get("dropout", 0.1), "seed": config.get("seed", 42),
                "drone_sample_boost": config.get("drone_sample_boost", 1.0),
                "bird_sample_boost": config.get("bird_sample_boost", 1.0),
                "balloon_sample_boost": config.get("balloon_sample_boost", 1.0),
                "clutter_sample_boost": config.get("clutter_sample_boost", 2.0),
                "unknown_sample_boost": config.get("unknown_sample_boost", 1.0),
                "augmentation_frame_drop_fraction": (config.get("trajectory_augmentation") or {}).get("frame_drop_fraction", config.get("augmentation_frame_drop_fraction", 0.10)),
                "augmentation_amplitude_scale_max_delta": (config.get("trajectory_augmentation") or {}).get("amplitude_scale_max_delta", config.get("augmentation_amplitude_scale_max_delta", 0.05)),
                "augmentation_snr_offset_db": (config.get("trajectory_augmentation") or {}).get("snr_offset_db", config.get("augmentation_snr_offset_db", 1.0)),
                "checkpoint_selection_metric": config.get("checkpoint_selection_metric", "macro_f1"),
                "class_weight_mode": config.get("class_weight_mode", "inverse_sqrt"),
                "class_balanced_beta": config.get("class_balanced_beta", 0.999),
                "class_weight_floor": config.get("class_weight_floor", 0.5),
                "class_weight_cap": config.get("class_weight_cap", 3.0),
                "manual_class_loss_weights": config.get("manual_class_loss_weights"),
                "use_cosface": config.get("use_cosface", True),
                "lr_scheduler": config.get("lr_scheduler", "cosine"),
                "skip_test": config.get("test_deferred", True),
                "partition_augmentation_diagnostics": (config.get("partition_augmentation_diagnostics") or {}).get("enabled", True),
                "partition_augmentation_method": config.get("partition_augmentation_method") or (config.get("trajectory_augmentation") or {}).get("method", "perturbation"),
                "partition_augmentation_targets_train": (config.get("trajectory_augmentation") or {}).get("targets", [489] * 5),
                "partition_augmentation_targets_val": ((config.get("partition_augmentation_diagnostics") or {}).get("val") or {}).get("targets", [105] * 5),
                "partition_augmentation_targets_test": ((config.get("partition_augmentation_diagnostics") or {}).get("test") or {}).get("targets", [105] * 5),
                "resume_checkpoint": str(checkpoint),
            }
            try:
                job = SCHEDULER.resume_in_place(old_name, ROOT, params)
            except (OSError, ValueError, TypeError) as exc:
                self.send_json(500, {"error": f"Failed to resume TR training: {exc}"})
                return
            invalidate_discover_cache()
            self.send_json(201, {"name": old_name, "queue_id": job["id"], "status": "queued", "source": old_name,
                                 "checkpoint": str(checkpoint), "resume_mode": resume_mode,
                                 "workers_adjusted": False, "workers": params["workers"],
                                 "resume_count": job.get("resume_count", 1), "in_place": True})
            return
        velocity = config.get("velocity_preprocessing") or {}
        interpolation = config.get("resampling") or velocity.get("interpolation") or "db_linear"
        interpolation = {"linear_in_db": "db_linear", "linear_in_power": "power_linear"}.get(interpolation, interpolation)
        augmentation = config.get("augmentation", "off")
        if isinstance(augmentation, dict):
            augmentation = augmentation.get("mode", "off")
        params = {
            "epochs": config.get("epochs", 50), "batch_size": config.get("batch_size", 128),
            "workers": config.get("workers", 4),
            "max_train_frames_per_trajectory": config.get("max_train_frames_per_trajectory", 32),
            "norm_samples": config.get("norm_samples", 2048), "learning_rate": config.get("learning_rate", 0.0003),
            "weight_decay": config.get("weight_decay", 0.0001), "patience": config.get("patience", 10),
            "seed": config.get("seed", 42), "velocity_min": config.get("velocity_min", -90),
            "velocity_max": config.get("velocity_max", 89),
            "target_width": config.get("target_width", velocity.get("target_width", 900)),
            "resampling": interpolation, "normalization": config.get("normalization", "global_z"),
            "input_mode": config.get("input_mode", "rd"), "model_head": config.get("model_head", "global"),
            "augmentation": augmentation,
            "split_mode": config.get("split", {}).get("mode", config.get("split_mode", "registry_train")),
            "train_registry": config.get("split", {}).get("registry_path", config.get("train_registry", "")),
            "grouped_split": config.get("split", {}).get("manifest", config.get("grouped_split", "")),
            "skip_test": config.get("skip_test", False), "resume_checkpoint": str(checkpoint),
            "partition_augmentation_diagnostics": (config.get("partition_augmentation_diagnostics") or {}).get("enabled", True),
            "partition_augmentation_targets_train": ((config.get("partition_augmentation_diagnostics") or {}).get("train") or {}).get("targets", [489] * 5),
            "partition_augmentation_targets_val": ((config.get("partition_augmentation_diagnostics") or {}).get("val") or {}).get("targets", [105] * 5),
            "partition_augmentation_targets_test": ((config.get("partition_augmentation_diagnostics") or {}).get("test") or {}).get("targets", [105] * 5),
        }
        # A resumed RD run must keep the same preprocessing cache contract as
        # its source experiment.  The older resume path accidentally omitted
        # this field, forcing every resumed epoch to reopen MAT files and
        # resample all 900-column RD frames.  Reuse a validated saved cache;
        # when an older config has lost that field, derive the compatible
        # default cache from the exact resumed protocol.
        configured_cache = str(config.get("rd_cache") or "").strip()
        if configured_cache:
            try:
                candidate_cache = Path(configured_cache).expanduser().resolve()
                params["rd_cache"] = (str(candidate_cache)
                                      if validate_rd_cache(candidate_cache, params) is None else "")
            except (OSError, ValueError, TypeError):
                params["rd_cache"] = ""
        if not params.get("rd_cache"):
            params["rd_cache"] = recommended_rd_cache(params)
        worker_adjusted = False
        error = current.get("error") or ""
        if "error code: <1455>" in error or "shared file mapping" in error.lower():
            params["workers"] = 0
            worker_adjusted = True
        dataset_root = Path(str(config.get("dataset_root") or DEFAULT_DATASET_ROOT)).expanduser().resolve()
        if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
            dataset_root = DEFAULT_DATASET_ROOT
        if not dataset_root.is_dir() or not (dataset_root / "MAT").is_dir():
            self.send_json(400, {"error": "Dataset root is unavailable"})
            return
        try:
            job = SCHEDULER.resume_in_place(old_name, dataset_root, params)
        except (OSError, ValueError, TypeError) as exc:
            self.send_json(500, {"error": f"Failed to resume training: {exc}"})
            return
        invalidate_discover_cache()
        self.send_json(201, {"name": old_name, "queue_id": job["id"], "status": "queued",
                             "source": old_name, "checkpoint": str(checkpoint),
                             "resume_mode": resume_mode, "workers_adjusted": worker_adjusted,
                             "workers": params["workers"], "resume_count": job.get("resume_count", 1),
                             "in_place": True})

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    SCHEDULER = TrainingScheduler(ROOT, resource_snapshot, powershell_processes)
    port = int(os.environ.get("RADAR_MONITOR_PORT", "8765"))
    host = os.environ.get("RADAR_MONITOR_HOST", "0.0.0.0")
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print(f"Training monitor: http://{display_host}:{port} (LAN enabled; use this PC's LAN IP from other devices)", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()

"""Persistent, resource-aware scheduler for local RadarInsight training jobs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path


OUTPUT_RE = re.compile(r"--output-dir\s+(?:\"([^\"]+)\"|(\S+))")
ACTIVE_STATES = {"starting", "running"}
TERMINAL_STATES = {"completed", "failed", "cancelled"}
CONFIRMATION_STATE = "pending_confirmation"


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


class TrainingScheduler:
    """Owns queued jobs while treating independently launched trainers as external load."""

    DEFAULT_SETTINGS = {
        "paused": False,
        "min_concurrency": 2,
        "max_concurrency": 2,
        "poll_seconds": 3.0,
        "hard_cpu_percent": 96.0,
        "hard_memory_percent": 90.0,
        "hard_gpu_memory_percent": 90.0,
        "expand_cpu_percent": 78.0,
        "expand_memory_percent": 78.0,
        "expand_gpu_memory_percent": 72.0,
    }

    def __init__(self, root, resource_provider, process_provider):
        self.root = Path(root).resolve()
        self.artifacts = self.root / "artifacts"
        self.state_path = self.root / "tmp" / "training_scheduler_state.json"
        self.snapshot_root = self.root / "tmp" / "training_code_snapshots"
        self.resource_provider = resource_provider
        self.process_provider = process_provider
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.children = {}
        self.state = self._load_state()
        self.thread = threading.Thread(target=self._loop, name="training-scheduler", daemon=True)
        self.thread.start()

    def _load_state(self):
        state = {"version": 1, "settings": copy.deepcopy(self.DEFAULT_SETTINGS), "jobs": [],
                 "last_error": None, "updated_at": iso_now()}
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
                settings = copy.deepcopy(self.DEFAULT_SETTINGS)
                settings.update(loaded.get("settings") or {})
                state["settings"] = settings
                state["jobs"] = loaded.get("jobs") if isinstance(loaded.get("jobs"), list) else []
        except (OSError, ValueError, TypeError):
            pass
        for index, job in enumerate(state["jobs"]):
            job.setdefault("order", index)
            job.setdefault("attempts", 0)
            job.setdefault("status", "queued")
        return state

    def _save_locked(self):
        self.state["updated_at"] = iso_now()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for _ in range(4):
            try:
                os.replace(str(temporary), str(self.state_path))
                return
            except PermissionError:
                time.sleep(0.05)
        os.replace(str(temporary), str(self.state_path))

    def _trainer_snapshot(self):
        source = self.root / "radar_rd" / "train.py"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        directory = self.snapshot_root / digest
        trainer = directory / "train.py"
        if not trainer.exists():
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(trainer))
            (directory / "snapshot.json").write_text(
                json.dumps({"sha256": digest, "source": str(source), "created_at": iso_now()},
                           ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return digest, trainer

    def _program_for(self, params):
        """Return the executable source and digest for one experiment type."""
        experiment_type = str(params.get("experiment_type") or "rd_only")
        if experiment_type == "rd_only":
            return self._trainer_snapshot()
        names = {
            "tr_only": "train_tr_ml.py" if str(params.get("tr_implementation", "dl")).lower() == "ml" else "train_tr_only.py",
            "tr_checkpoint_eval": "evaluate_tr_only.py",
            "rd_checkpoint_eval": "evaluate_rd_only.py",
            "tr_rd_soft_cascade": "evaluate_soft_cascade.py",
            "fusion_gate_training": "generate_fusion_oof.py",
            "fusion_gate_calibration": "train_calibration_gate.py",
        }
        try:
            source = self.root / "scripts" / names[experiment_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported experiment type: {experiment_type}") from exc
        if not source.is_file():
            raise ValueError(f"Experiment program is missing: {source}")
        return hashlib.sha256(source.read_bytes()).hexdigest(), source

    def _python_executable(self):
        preferred = Path(r"C:\Users\Surfa\AppData\Local\Programs\Python\Python39\python.exe")
        return preferred if preferred.exists() else Path(sys.executable)

    def _command_for(self, job):
        params = job["params"]
        def partition_augmentation_args(include_train_switch=False, include_method=False):
            # Train-target expansion is an independent training configuration.
            # The diagnostics toggle applies only to validation/test virtual
            # tracks; omitting train targets made the trainer fall back to
            # “all classes up to the maximum”, silently changing a baseline.
            command = (["--partition-augmentation-diagnostics"]
                       if params.get("partition_augmentation_diagnostics", True)
                       else ["--no-partition-augmentation-diagnostics"])
            if include_method:
                command.extend(["--partition-augmentation-method",
                                str(params.get("partition_augmentation_method", "perturbation"))])
            train_enabled = params.get("partition_augmentation_train_enabled")
            if include_train_switch and train_enabled is not None:
                command.extend(["--partition-augmentation-train-enabled",
                                *["1" if bool(value) else "0" for value in train_enabled]])
            target_defaults = {"train": [489] * 5, "val": [105] * 5, "test": [105] * 5}
            for partition in ("train", "val", "test"):
                values = params.get(f"partition_augmentation_targets_{partition}", target_defaults[partition])
                if values is not None and (partition == "train" or params.get("partition_augmentation_diagnostics", True)):
                    command.extend([f"--partition-augmentation-targets-{partition}", *[str(value) for value in values]])
            return command
        experiment_type = str(params.get("experiment_type") or "rd_only")
        if experiment_type == "tr_only":
            if str(params.get("tr_implementation", "dl")).lower() == "ml":
                command = [
                    str(self._python_executable()), str(job["snapshot_path"]),
                    "--output-dir", job["output_dir"], "--split", str(params["grouped_split"]),
                    "--track-index", str(params["track_index"]), "--seed", str(params.get("seed", 42)),
                    "--models", *[str(value) for value in params.get("ml_models", ["lightgbm", "hist_gradient_boosting"])],
                ]
                ml_weights = params.get("ml_model_weights")
                if ml_weights:
                    command.extend(["--model-weights", *[str(value) for value in ml_weights]])
                command.extend(["--model-plan", str(params.get("ml_model_plan", "custom"))])
                if params.get("ml_soft_voting", False): command.append("--soft-voting")
                if params.get("skip_test", True): command.append("--skip-test")
                return command
            command = [
                str(self._python_executable()), str(job["snapshot_path"]),
                "--output-dir", job["output_dir"],
                "--split", str(params["grouped_split"]),
                "--track-index", str(params["track_index"]),
                "--epochs", str(params.get("epochs", 24)),
                "--batch-size", str(params.get("batch_size_tr", params.get("batch_size", 8))),
                "--workers", str(params.get("workers", 0)),
                "--learning-rate", str(params.get("learning_rate", 0.0002)),
                "--weight-decay", str(params.get("weight_decay", 0.0001)),
                "--lr-scheduler", str(params.get("lr_scheduler", "none")),
                "--patience", str(params.get("patience", 0)),
                "--dropout", str(params.get("dropout", 0.1)),
                "--seed", str(params.get("seed", 42)),
                "--sampling-mode", str(params.get("sampling_mode", "b01_balanced")),
                "--sampling-protocol", str(params.get("sampling_protocol", "coverage_plus_boost")),
                "--drone-sample-boost", str(params.get("drone_sample_boost", 1.0)),
                "--bird-sample-boost", str(params.get("bird_sample_boost", 1.0)),
                "--balloon-sample-boost", str(params.get("balloon_sample_boost", 1.0)),
                "--unknown-sample-boost", str(params.get("unknown_sample_boost", 1.0)),
                "--clutter-sample-boost", str(params.get("clutter_sample_boost", 2.0)),
                "--augmentation-frame-drop-fraction", str(params.get("augmentation_frame_drop_fraction", 0.10)),
                "--augmentation-amplitude-scale-max-delta", str(params.get("augmentation_amplitude_scale_max_delta", 0.05)),
                "--augmentation-snr-offset-db", str(params.get("augmentation_snr_offset_db", 1.0)),
                "--checkpoint-selection-metric", str(params.get("checkpoint_selection_metric", "macro_f1")),
                "--class-weight-mode", str(params.get("class_weight_mode", "inverse_sqrt")),
                "--class-balanced-beta", str(params.get("class_balanced_beta", 0.999)),
                "--class-weight-floor", str(params.get("class_weight_floor", 0.25)),
                "--class-weight-cap", str(params.get("class_weight_cap", 2.0)),
                "--sampling-class-enabled", *[str(int(value)) for value in params.get("sampling_class_enabled", [True] * 5)],
                "--class-loss-enabled", *[str(int(value)) for value in params.get("class_loss_enabled", [True] * 5)],
            ] + partition_augmentation_args(include_train_switch=True, include_method=True) + (["--skip-test"] if params.get("skip_test", True) else [])
            command.append("--use-cosface" if params.get("use_cosface", True) else "--no-use-cosface")
            command.append("--manual-sampling-boosts-enabled" if params.get("manual_sampling_boosts_enabled", False)
                           else "--no-manual-sampling-boosts-enabled")
            manual_class_loss_weights = params.get("manual_class_loss_weights")
            if manual_class_loss_weights is not None:
                command.extend(["--manual-class-loss-weights", *[str(value) for value in manual_class_loss_weights]])
            if params.get("resume_checkpoint"):
                command.extend(["--resume", str(params["resume_checkpoint"])])
            return command
        if experiment_type == "tr_checkpoint_eval":
            command = [
                str(self._python_executable()), str(job["snapshot_path"]),
                "--output-dir", job["output_dir"],
                "--split", str(params["grouped_split"]),
                "--track-index", str(params["track_index"]),
                "--checkpoint", str(params["tr_checkpoint"]),
                "--partition", str(params.get("partition", "val")),
                "--batch-size", str(params.get("batch_size_tr", 32)),
                "--workers", str(params.get("workers", 0)),
            ] + partition_augmentation_args(include_method=True)
            if params.get("augment_existing"):
                command.append("--augment-existing")
            return command
        if experiment_type == "rd_checkpoint_eval":
            command = [
                str(self._python_executable()), str(job["snapshot_path"]),
                "--dataset-root", job["dataset_root"],
                "--output-dir", job["output_dir"],
                "--checkpoint", str(params["rd_checkpoint"]),
                "--partition", str(params.get("partition", "val")),
                "--batch-size", str(params.get("batch_size_rd", 128)),
                "--workers", str(params.get("workers", 0)),
            ]
            if params.get("rd_cache"):
                command.extend(["--rd-cache", str(params["rd_cache"])])
            command.extend(partition_augmentation_args())
            if params.get("augment_existing"):
                command.append("--augment-existing")
            return command
        if experiment_type == "tr_rd_soft_cascade":
            command = [
                str(self._python_executable()), str(job["snapshot_path"]),
                "--dataset-root", job["dataset_root"],
                "--output-dir", job["output_dir"],
                "--split", str(params["grouped_split"]),
                "--track-index", str(params["track_index"]),
                "--tr-checkpoint", str(params["tr_checkpoint"]),
                "--rd-checkpoint", str(params["rd_checkpoint"]),
                "--partition", str(params.get("partition", "val")),
                "--batch-size-tr", str(params.get("batch_size_tr", 32)),
                "--batch-size-rd", str(params.get("batch_size_rd", 128)),
                "--workers", str(params.get("workers", 0)),
                "--fusion-mode", str(params.get("fusion_mode", "fixed")),
            ]
            weights = params.get("fixed_rd_weight", [0.2])
            if not isinstance(weights, list):
                weights = [weights]
            if not weights:
                weights = [0.2]
            command.append("--fixed-rd-weight")
            command.extend(str(value) for value in weights)
            if params.get("rd_cache"):
                command.extend(["--rd-cache", str(params["rd_cache"])])
            if params.get("fusion_checkpoint"):
                command.extend(["--fusion-checkpoint", str(params["fusion_checkpoint"])])
            if str(params.get("partition", "val")).lower() == "test":
                command.append("--allow-test")
            command.extend(partition_augmentation_args(include_method=True))
            if params.get("augment_existing"):
                command.append("--augment-existing")
            return command
        if experiment_type == "fusion_gate_training":
            command = [
                str(self._python_executable()), str(job["snapshot_path"]),
                "--dataset-root", job["dataset_root"],
                "--output-dir", job["output_dir"],
                "--split", str(params["grouped_split"]),
                "--track-index", str(params["track_index"]),
                "--tr-template-checkpoint", str(params["tr_checkpoint"]),
                "--rd-template-checkpoint", str(params["rd_checkpoint"]),
                "--folds", str(params.get("folds", 5)),
                "--inner-val-fraction", str(params.get("inner_val_fraction", 0.15)),
                "--tr-epochs", str(params.get("tr_epochs", 0)),
                "--rd-epochs", str(params.get("rd_epochs", 0)),
                "--batch-size-tr", str(params.get("batch_size_tr", 8)),
                "--batch-size-rd", str(params.get("batch_size_rd", 128)),
                "--workers", str(params.get("workers", 0)),
                "--gate-epochs", str(params.get("gate_epochs", 200)),
                "--gate-learning-rate", str(params.get("gate_learning_rate", 0.01)),
                "--gate-weight-decay", str(params.get("gate_weight_decay", 0.0001)),
                "--gate-hidden-dim", str(params.get("gate_hidden_dim", 32)),
                "--initial-rd-weight", str(params.get("initial_rd_weight", 0.2)),
                "--seed", str(params.get("seed", 42)),
            ]
            if params.get("rd_cache"):
                command.extend(["--rd-cache", str(params["rd_cache"])])
            return command
        if experiment_type == "fusion_gate_calibration":
            command = [
                str(self._python_executable()), str(job["snapshot_path"]),
                "--dataset-root", job["dataset_root"],
                "--output-dir", job["output_dir"],
                "--split", str(params["grouped_split"]),
                "--track-index", str(params["track_index"]),
                "--tr-checkpoint", str(params["tr_checkpoint"]),
                "--rd-checkpoint", str(params["rd_checkpoint"]),
                "--calibration-partition", str(params.get("calibration_partition", "val")),
                "--batch-size-tr", str(params.get("batch_size_tr", 8)),
                "--batch-size-rd", str(params.get("batch_size_rd", 128)),
                "--workers", str(params.get("workers", 0)),
                "--gate-epochs", str(params.get("gate_epochs", 200)),
                "--gate-learning-rate", str(params.get("gate_learning_rate", 0.01)),
                "--gate-weight-decay", str(params.get("gate_weight_decay", 0.0001)),
                "--gate-hidden-dim", str(params.get("gate_hidden_dim", 32)),
                "--initial-rd-weight", str(params.get("initial_rd_weight", 0.2)),
                "--seed", str(params.get("seed", 42)),
            ]
            if params.get("rd_cache"):
                command.extend(["--rd-cache", str(params["rd_cache"])])
            return command
        command = [
            str(self._python_executable()), str(job["snapshot_path"]),
            "--dataset-root", job["dataset_root"], "--output-dir", job["output_dir"],
            "--epochs", str(params["epochs"]), "--batch-size", str(params["batch_size"]),
            "--workers", str(params["workers"]),
            "--max-train-frames-per-trajectory", str(params["max_train_frames_per_trajectory"]),
            "--norm-samples", str(params["norm_samples"]),
            "--learning-rate", str(params["learning_rate"]),
            "--weight-decay", str(params["weight_decay"]),
            "--patience", str(params["patience"]), "--seed", str(params["seed"]),
            "--velocity-min", str(params["velocity_min"]),
            "--velocity-max", str(params["velocity_max"]),
            "--target-width", str(params["target_width"]),
            "--resampling", str(params["resampling"]),
            "--normalization", str(params["normalization"]),
            "--input-mode", str(params["input_mode"]),
            "--model-head", str(params.get("model_head", "global")),
            "--augmentation", str(params.get("augmentation", "off")),
            "--split-mode", str(params.get("split_mode", "registry_train")),
        ]
        # ``grouped_split`` owns all three partitions.  It must take
        # precedence over the legacy train-only registry so RD validation,
        # test, TR and fusion use the exact same trajectory boundaries.
        command.extend(partition_augmentation_args())
        if params.get("grouped_split"):
            command.extend(["--grouped-split", str(params["grouped_split"])])
        if params.get("train_registry"):
            command.extend(["--train-registry", str(params["train_registry"])])
        if params.get("rd_cache"):
            command.extend(["--rd-cache", str(params["rd_cache"])])
        if params.get("resume_checkpoint"):
            command.extend(["--resume", str(params["resume_checkpoint"])])
        if params.get("skip_test"):
            command.append("--skip-test")
        return command

    def enqueue(self, name, dataset_root, params):
        digest, trainer = self._program_for(params)
        output = (self.artifacts / name).resolve()
        job = {
            "id": uuid.uuid4().hex,
            "name": name,
            # Creating an experiment must never consume GPU resources by
            # itself.  A user explicitly promotes it into the runnable queue.
            "status": CONFIRMATION_STATE,
            "order": 0,
            "created_at": iso_now(),
            "queued_at": None,
            "confirmed_at": None,
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "attempts": 0,
            "dataset_root": str(dataset_root),
            "output_dir": str(output),
            "log_path": str(self.artifacts / f"{name}.log"),
            "error_log_path": str(self.artifacts / f"{name}.err.log"),
            "params": copy.deepcopy(params),
            "code_hash": digest,
            "snapshot_path": str(trainer),
            "command": [],
            "error": None,
        }
        job["command"] = self._command_for(job)
        with self.lock:
            if any(item.get("name") == name and item.get("status") not in {"cancelled"} for item in self.state["jobs"]):
                raise ValueError("An experiment with this name is already in the queue")
            job["order"] = 1 + max([item.get("order", 0) for item in self.state["jobs"]] or [-1])
            self.state["jobs"].append(job)
            self._save_locked()
        return copy.deepcopy(job)

    def resume_in_place(self, name, dataset_root, params):
        """Requeue a terminal training attempt without changing its identity.

        A resume is a continuation of the same experiment, not a new
        experiment.  Keeping the original scheduler record also keeps the
        output directory, chronological experiment number and audit trail
        stable.  ``params`` must contain the checkpoint in that output
        directory; ``_start_job`` validates the same constraint before it
        permits reuse of an existing output directory.
        """
        checkpoint = Path(str(params.get("resume_checkpoint") or "")).expanduser().resolve()
        with self.lock:
            matches = [job for job in self.state["jobs"] if job.get("name") == name]
            job = matches[-1] if matches else None
            if job is None:
                raise ValueError("This experiment is not managed by the scheduler and cannot resume in place")
            if job.get("status") not in {"failed", "cancelled", "incomplete"}:
                raise ValueError("Only failed, incomplete, or cancelled experiments can resume")
            output = Path(job.get("output_dir", "")).resolve()
            if checkpoint.parent != output or not checkpoint.is_file():
                raise ValueError("Resume checkpoint must be last.pt or best.pt in the original output directory")
            now = iso_now()
            job.update({
                "status": "queued", "queued_at": now, "confirmed_at": now,
                "started_at": None, "finished_at": None, "pid": None,
                "error": None, "dataset_root": str(dataset_root),
                "params": copy.deepcopy(params), "command": [],
                "resume_count": int(job.get("resume_count") or 0) + 1,
                "resume_checkpoint": str(checkpoint),
                "resume_requested_at": now,
            })
            # A resumed job goes behind any already-approved work, but keeps
            # its original ``created_at`` and output identity for numbering.
            job["order"] = 1 + max([item.get("order", 0) for item in self.state["jobs"]] or [-1])
            job["command"] = self._command_for(job)
            self._save_locked()
            return copy.deepcopy(job)

    def supplement_augmentation_in_place(self, name, dataset_root, params):
        """Append a partition-local augmentation diagnostic to a completed evaluation.

        The scheduler record, output directory, experiment number and original
        timestamps remain unchanged.  Only the evaluation command is replaced
        with its explicit ``--augment-existing`` variant and queued immediately.
        """
        with self.lock:
            matches = [job for job in self.state["jobs"] if job.get("name") == name]
            job = matches[-1] if matches else None
            if job is None:
                # Older completed evaluations may predate the persistent
                # scheduler registry.  Adopt their existing artifact in place
                # instead of creating a second experiment identity.
                output = (self.artifacts / name).resolve()
                if not output.is_dir():
                    raise ValueError("Original evaluation output directory does not exist")
                digest, program = self._program_for(params)
                try:
                    created_at = datetime.fromtimestamp(output.stat().st_ctime).isoformat(timespec="seconds")
                except OSError:
                    created_at = iso_now()
                job = {
                    "id": uuid.uuid4().hex, "name": name, "status": "completed",
                    "order": 1 + max([item.get("order", 0) for item in self.state["jobs"]] or [-1]),
                    "created_at": created_at, "queued_at": None, "confirmed_at": None,
                    "started_at": None, "finished_at": created_at, "pid": None,
                    "attempts": 1, "dataset_root": str(dataset_root),
                    "output_dir": str(output), "log_path": str(self.artifacts / f"{name}.log"),
                    "error_log_path": str(self.artifacts / f"{name}.err.log"),
                    "params": copy.deepcopy(params), "code_hash": digest, "snapshot_path": str(program),
                    "command": [], "error": None, "adopted_for_supplement": True,
                }
                self.state["jobs"].append(job)
            if job is None:
                raise ValueError("This experiment is not managed by the scheduler")
            if job.get("status") != "completed":
                raise ValueError("Only completed evaluations can receive an augmentation diagnostic")
            experiment_type = str((job.get("params") or {}).get("experiment_type") or "")
            if experiment_type not in {"rd_checkpoint_eval", "tr_checkpoint_eval", "tr_rd_soft_cascade"}:
                raise ValueError("Only RD, TR, or fusion evaluations support augmentation supplements")
            output = Path(job.get("output_dir", "")).resolve()
            if not output.is_dir():
                raise ValueError("Original evaluation output directory does not exist")
            if any((output / filename).is_file() for filename in (
                "test_augmented_trajectory_metrics.json", "validation_augmented_best.json",
                "augmented_metrics.json", "trajectory_decisions_augmented.jsonl",
                "trajectory_decisions_test_augmented.jsonl")):
                raise ValueError("This evaluation already has augmentation diagnostic results")
            now = iso_now()
            merged = copy.deepcopy(params)
            merged["experiment_type"] = experiment_type
            merged["augment_existing"] = True
            merged["partition_augmentation_diagnostics"] = True
            job.update({"status": "queued", "queued_at": now, "confirmed_at": now,
                        "started_at": None, "finished_at": None, "pid": None,
                        "error": None, "dataset_root": str(dataset_root),
                        "params": merged, "command": [],
                        "augmentation_supplement_count": int(job.get("augmentation_supplement_count") or 0) + 1,
                        "augmentation_supplement_requested_at": now})
            job["order"] = 1 + max([item.get("order", 0) for item in self.state["jobs"]] or [-1])
            job["command"] = self._command_for(job)
            self._save_locked()
            return copy.deepcopy(job)

    def has_name(self, name):
        with self.lock:
            return any(job.get("name") == name and job.get("status") not in {"cancelled"}
                       for job in self.state["jobs"])

    def _ordered_pending_locked(self):
        return sorted((job for job in self.state["jobs"] if job.get("status") == "queued"),
                      key=lambda item: (item.get("order", 0), item.get("created_at", "")))

    def confirm(self, job_id):
        """Promote a reviewed draft into the scheduler's runnable queue."""
        with self.lock:
            job = next((item for item in self.state["jobs"] if item.get("id") == job_id), None)
            if not job or job.get("status") != CONFIRMATION_STATE:
                raise ValueError("Only experiments awaiting confirmation can be added to the run queue")
            now = iso_now()
            job.update({"status": "queued", "queued_at": now, "confirmed_at": now, "error": None})
            self._save_locked()
            return copy.deepcopy(job)

    def update_settings(self, values):
        allowed = set(self.DEFAULT_SETTINGS)
        with self.lock:
            settings = copy.deepcopy(self.state["settings"])
            for key, value in values.items():
                if key not in allowed:
                    continue
                if key == "paused":
                    settings[key] = bool(value)
                elif key in {"min_concurrency", "max_concurrency"}:
                    settings[key] = int(value)
                else:
                    settings[key] = float(value)
            if not 1 <= settings["min_concurrency"] <= 8:
                raise ValueError("Minimum concurrency must be between 1 and 8")
            if not settings["min_concurrency"] <= settings["max_concurrency"] <= 8:
                raise ValueError("Maximum concurrency must be between minimum concurrency and 8")
            for key in allowed - {"paused", "min_concurrency", "max_concurrency", "poll_seconds"}:
                if not 0 <= settings[key] <= 100:
                    raise ValueError("Resource thresholds must be between 0 and 100")
            if not 1 <= settings["poll_seconds"] <= 60:
                raise ValueError("Polling interval must be between 1 and 60 seconds")
            self.state["settings"] = settings
            self._save_locked()
            return copy.deepcopy(settings)

    def move(self, job_id, direction):
        with self.lock:
            pending = self._ordered_pending_locked()
            index = next((i for i, job in enumerate(pending) if job.get("id") == job_id), None)
            if index is None:
                raise ValueError("Only queued experiments can be reordered")
            if direction == "top":
                target = 0
            elif direction == "up":
                target = max(0, index - 1)
            elif direction == "down":
                target = min(len(pending) - 1, index + 1)
            else:
                raise ValueError("Direction must be top, up, or down")
            job = pending.pop(index)
            pending.insert(target, job)
            for order, item in enumerate(pending):
                item["order"] = order
            self._save_locked()

    def cancel(self, job_id):
        """Cancel a queued or running scheduler-owned job."""
        with self.lock:
            job = next((item for item in self.state["jobs"] if item.get("id") == job_id), None)
            if not job or job.get("status") not in {CONFIRMATION_STATE, "queued", "starting", "running"}:
                raise ValueError("Only pending, queued, or running experiments can be cancelled")
            status = job.get("status")
            process = self.children.get(job_id)
            pid = job.get("pid") or (process.pid if process is not None else None)
            if status in {"starting", "running"}:
                if pid:
                    # taskkill /T also stops DataLoader worker descendants.
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                elif process is not None and process.poll() is None:
                    process.terminate()
                job.update({"status": "cancelled", "pid": None, "finished_at": iso_now(),
                            "error": "Cancelled by user"})
                self.children.pop(job_id, None)
            else:
                message = "Cancelled before confirmation" if status == CONFIRMATION_STATE else "Cancelled before start"
                job.update({"status": "cancelled", "finished_at": iso_now(), "error": message})
            self._save_locked()

    def rename(self, old_name, new_name):
        with self.lock:
            job = next((item for item in self.state["jobs"] if item.get("name") == old_name), None)
            if not job:
                return False
            if job.get("status") in ACTIVE_STATES:
                raise ValueError("Cannot rename a running experiment")
            if job.get("status") in {CONFIRMATION_STATE, "queued"}:
                job["name"] = new_name
                job["output_dir"] = str((self.artifacts / new_name).resolve())
                job["log_path"] = str(self.artifacts / f"{new_name}.log")
                job["error_log_path"] = str(self.artifacts / f"{new_name}.err.log")
                job["command"] = self._command_for(job)
            else:
                job["name"] = new_name
                job["output_dir"] = str((self.artifacts / new_name).resolve())
                job["log_path"] = str(self.artifacts / f"{new_name}.log")
                job["error_log_path"] = str(self.artifacts / f"{new_name}.err.log")
            self._save_locked()
            return True

    def snapshot(self):
        processes = self._process_map()
        with self.lock:
            state = copy.deepcopy(self.state)
        active_outputs = {str(Path(job.get("output_dir", "")).resolve()) for job in state["jobs"]
                          if job.get("status") in ACTIVE_STATES}
        external = [value for key, value in processes.items() if key not in active_outputs]
        pending = sorted((job for job in state["jobs"] if job.get("status") == "queued"),
                         key=lambda item: (item.get("order", 0), item.get("created_at", "")))
        positions = {job["id"]: index + 1 for index, job in enumerate(pending)}
        for job in state["jobs"]:
            job["queue_position"] = positions.get(job.get("id"))
        state["summary"] = {
            "awaiting_confirmation": sum(job.get("status") == CONFIRMATION_STATE for job in state["jobs"]),
            "queued": len(pending),
            "running": sum(job.get("status") in ACTIVE_STATES for job in state["jobs"]),
            "completed": sum(job.get("status") == "completed" for job in state["jobs"]),
            "failed": sum(job.get("status") == "failed" for job in state["jobs"]),
            "external_running": len(external),
        }
        return state

    def experiment_records(self):
        state = self.snapshot()
        records = []
        for job in state["jobs"]:
            records.append({
                "queue_id": job.get("id"), "queue_status": job.get("status"),
                "queue_position": job.get("queue_position"), "queue_order": job.get("order"),
                "code_hash": job.get("code_hash"),
                "name": job.get("name"), "output_dir": job.get("output_dir"),
                "pid": job.get("pid"), "created_at": job.get("created_at"),
                "started_at": job.get("started_at"), "finished_at": job.get("finished_at"),
                "resume_count": int(job.get("resume_count") or 0),
                "resume_checkpoint": job.get("resume_checkpoint"),
                "resume_requested_at": job.get("resume_requested_at"),
                "queue_error": job.get("error"), "params": copy.deepcopy(job.get("params") or {}),
            })
        return records

    def _process_map(self):
        result = {}
        try:
            processes = self.process_provider()
        except Exception:
            return result
        for process in processes:
            command = process.get("CommandLine") or ""
            if ("radar_rd.train" not in command and "training_code_snapshots" not in command
                    and "train_tr_only.py" not in command and "evaluate_tr_only.py" not in command
                    and "evaluate_soft_cascade.py" not in command
                    and "generate_fusion_oof.py" not in command
                    and "train_calibration_gate.py" not in command):
                continue
            match = OUTPUT_RE.search(command)
            if not match:
                continue
            output = Path(match.group(1) or match.group(2))
            if not output.is_absolute():
                output = self.root / output
            result[str(output.resolve())] = {
                "pid": int(process.get("ProcessId")), "command": command,
            }
        return result

    def _completion_exists(self, output):
        # RD/TR training writes ablation_complete.json or test metrics, while
        # a successful TR-RD soft-cascade evaluation writes metrics.json.
        return (
            (output / "ablation_complete.json").exists()
            or (output / "test_trajectory_metrics.json").exists()
            or (output / "metrics.json").exists()
        )

    def _tail_error(self, job):
        path = Path(job["error_log_path"])
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            return text[-4000:] if text else "Training stopped without a completion marker"
        except OSError:
            return "Training stopped without a completion marker"

    def _reconcile(self, processes):
        changed = False
        with self.lock:
            for job in self.state["jobs"]:
                output = Path(job["output_dir"])
                # Repair older soft-cascade evaluations that completed before
                # metrics.json was recognized as a completion artifact.
                if job.get("status") == "failed" and not (job.get("params") or {}).get("augment_existing") and self._completion_exists(output):
                    job.update({"status": "completed", "pid": None, "finished_at": iso_now(), "error": None})
                    changed = True
                    continue
                if job.get("status") not in ACTIVE_STATES:
                    continue
                handle = self.children.get(job["id"])
                if handle is not None and handle.poll() is None:
                    if job.get("status") != "running" or job.get("pid") != handle.pid:
                        job.update({"status": "running", "pid": handle.pid})
                        changed = True
                    continue
                live = processes.get(str(output.resolve()))
                if live:
                    if job.get("status") != "running" or job.get("pid") != live["pid"]:
                        job.update({"status": "running", "pid": live["pid"]})
                        changed = True
                    continue
                supplement = bool((job.get("params") or {}).get("augment_existing"))
                supplement_complete = (output / "augmentation_supplement.json").is_file()
                if (supplement and supplement_complete) or (not supplement and self._completion_exists(output)):
                    job.update({"status": "completed", "pid": None, "finished_at": iso_now(), "error": None})
                else:
                    exit_code = handle.returncode if handle is not None else None
                    message = self._tail_error(job)
                    if exit_code not in (None, 0):
                        message = f"Exit code {exit_code}: {message}"
                    job.update({"status": "failed", "pid": None, "finished_at": iso_now(), "error": message})
                self.children.pop(job["id"], None)
                changed = True
            if changed:
                self._save_locked()

    @staticmethod
    def _percent(value, fallback=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _target_concurrency(self, resources, total_active, has_pending):
        settings = self.state["settings"]
        if settings.get("paused") or not has_pending:
            return total_active, "paused" if settings.get("paused") else "no pending jobs"
        system = (resources or {}).get("system") or {}
        gpu = system.get("gpu") or {}
        cpu = self._percent(system.get("cpu_percent"))
        memory = self._percent(system.get("memory_percent"))
        gpu_memory = self._percent(gpu.get("memory_percent")) if gpu.get("available") else 0.0
        hard = (cpu >= settings["hard_cpu_percent"] or memory >= settings["hard_memory_percent"] or
                gpu_memory >= settings["hard_gpu_memory_percent"])
        if hard:
            return total_active, "resource hard limit"
        target = settings["min_concurrency"]
        expandable = (cpu <= settings["expand_cpu_percent"] and memory <= settings["expand_memory_percent"] and
                      (not gpu.get("available") or gpu_memory <= settings["expand_gpu_memory_percent"]))
        if expandable:
            target = settings["max_concurrency"]
            # Estimate the next trainer's VRAM from current aggregate usage.
            # This is deliberately conservative because nvidia-smi includes
            # desktop allocations in memory.used and a new CUDA context has a
            # short startup spike before per-process telemetry stabilizes.
            if gpu.get("available") and total_active > 0:
                estimated_per_job = max(gpu_memory / total_active, 8.0)
                safe_total = int(settings["hard_gpu_memory_percent"] // estimated_per_job)
                target = min(target, max(settings["min_concurrency"], safe_total))
        return max(total_active, target), "expanded" if expandable else "minimum concurrency"

    def _start_job(self, job_id):
        with self.lock:
            job = next((item for item in self.state["jobs"] if item.get("id") == job_id), None)
            if not job or job.get("status") != "queued":
                return
            output = Path(job["output_dir"])
            log_path = Path(job["log_path"])
            error_path = Path(job["error_log_path"])
            resume_checkpoint = (job.get("params") or {}).get("resume_checkpoint")
            is_in_place_supplement = bool((job.get("params") or {}).get("augment_existing"))
            try:
                is_in_place_resume = bool(resume_checkpoint and Path(str(resume_checkpoint)).expanduser().resolve().parent == output.resolve())
            except (OSError, ValueError):
                is_in_place_resume = False
            if (output.exists() or log_path.exists() or error_path.exists()) and not (is_in_place_resume or is_in_place_supplement):
                job.update({"status": "failed", "finished_at": iso_now(),
                            "error": "Output directory or log already exists before scheduler start"})
                self._save_locked()
                return
            job.update({"status": "starting", "started_at": iso_now(), "attempts": job.get("attempts", 0) + 1,
                        "error": None})
            self._save_locked()
        try:
            output.mkdir(parents=True, exist_ok=(is_in_place_resume or is_in_place_supplement))
            log_mode = "a" if (is_in_place_resume or is_in_place_supplement) else "w"
            with log_path.open(log_mode, encoding="utf-8") as stdout, error_path.open(log_mode, encoding="utf-8") as stderr:
                if is_in_place_resume:
                    stdout.write(f"\n--- resume attempt {job.get('resume_count', 1)} at {iso_now()} ---\n")
                    stdout.flush()
                elif is_in_place_supplement:
                    stdout.write(f"\n--- augmentation diagnostic supplement {job.get('augmentation_supplement_count', 1)} at {iso_now()} ---\n")
                    stdout.flush()
                process_env = os.environ.copy()
                existing_pythonpath = process_env.get("PYTHONPATH", "")
                process_env["PYTHONPATH"] = (
                    str(self.root) if not existing_pythonpath
                    else str(self.root) + os.pathsep + existing_pythonpath
                )
                process = subprocess.Popen(
                    job["command"], cwd=str(self.root), stdout=stdout, stderr=stderr,
                    env=process_env,
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                                  getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            with self.lock:
                job.update({"status": "running", "pid": process.pid})
                self.children[job_id] = process
                self._save_locked()
        except OSError as exc:
            with self.lock:
                job.update({"status": "failed", "pid": None, "finished_at": iso_now(), "error": str(exc)})
                self._save_locked()

    @staticmethod
    def _cache_ready(job):
        cache = (job.get("params") or {}).get("rd_cache")
        if not cache:
            return True
        cache_dir = Path(str(cache))
        return (cache_dir / "complete.json").is_file() and (cache_dir / "metadata.json").is_file() and (cache_dir / "index.json").is_file()

    def _tick(self):
        processes = self._process_map()
        self._reconcile(processes)
        try:
            resources = self.resource_provider()
        except Exception:
            resources = {}
        with self.lock:
            pending = self._ordered_pending_locked()
            managed_outputs = {str(Path(job["output_dir"]).resolve()) for job in self.state["jobs"]
                               if job.get("status") in ACTIVE_STATES}
            managed_active = len(managed_outputs)
            external_active = sum(output not in managed_outputs for output in processes)
            total_active = managed_active + external_active
            target, reason = self._target_concurrency(resources, total_active, bool(pending))
            runnable = [job for job in pending if self._cache_ready(job)]
            capacity = max(0, min(len(runnable), target - total_active))
            start_ids = [job["id"] for job in runnable[:capacity]]
            self.state["decision"] = {
                "at": iso_now(), "reason": reason, "target_concurrency": target,
                "managed_active": managed_active, "external_active": external_active,
                "starting": len(start_ids), "waiting_for_cache": len(pending) - len(runnable),
            }
            self.state["last_error"] = None
            self._save_locked()
        for job_id in start_ids:
            self._start_job(job_id)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                with self.lock:
                    self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                    self._save_locked()
            with self.lock:
                seconds = float(self.state["settings"].get("poll_seconds", 3.0))
            self.stop_event.wait(max(1.0, seconds))

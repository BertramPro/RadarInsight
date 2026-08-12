import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from training_monitor_server import (  # noqa: E402
    DEFAULT_F_SPLIT_RD_CACHE, DEFAULT_GROUPED_SPLIT, automatic_tr_balance_preview,
    recommended_rd_cache, validate_rd_cache,
    Handler, experiment_interface, read_trajectory_decisions,
)
from training_scheduler import TrainingScheduler  # noqa: E402
from radar_fusion.partition_augmentation import expand_rd_frames  # noqa: E402
from radar_rd.train import Frame  # noqa: E402


class ExperimentInterfaceTests(unittest.TestCase):
    def test_checkpoint_catalog_refresh_preserves_live_gate_calibration_selection(self):
        html = (ROOT / "scripts" / "training_monitor.html").read_text(encoding="utf-8")
        self.assertIn("checkpointSelectionRevision.tr===revisionAtStart.tr", html)
        self.assertIn("checkpointSelectionRevision.rd===revisionAtStart.rd", html)
        self.assertIn("checkpointSelectionRevision.gate===revisionAtStart.gate", html)
        self.assertIn("experiment_type')?.value!=='tr_rd_soft_cascade'", html)
        self.assertIn("checkpointSelectionRevision[kind]++", html)

    def test_tr_cosface_toggle_keeps_live_training_curves(self):
        html = (ROOT / "scripts" / "training_monitor.html").read_text(encoding="utf-8")
        self.assertIn('name="use_cosface"', html)
        self.assertIn("payload.use_cosface=!!form.elements.use_cosface?.checked", html)
        self.assertIn("r.train_frame_accuracy", html)
        self.assertIn("CosFace 已开启", html)
        self.assertIn("CosFace 已关闭", html)

    @patch("training_monitor_server.tr_train_base_counts", return_value=[489, 245, 245, 35, 70])
    def test_tr_balance_preview_exposes_each_combination_stage(self, _counts):
        preview = automatic_tr_balance_preview(
            "split.json", "tracks.csv", [489, 245, 245, 35, 70],
            enabled=[True] * 5, sampling_enabled=[True] * 5, loss_enabled=[True] * 5,
            sampling_mode="b01_balanced", class_weight_mode="inverse_sqrt",
            beta=0.999, floor=0.25, cap=2.0,
        )
        self.assertEqual(preview["effective_counts"], [489, 245, 245, 35, 70])
        self.assertAlmostEqual(sum(preview["class_sampling_probability"]), 1.0)
        for sample, loss, raw, final in zip(
            preview["sampling_effective_weights"], preview["class_loss_weights"],
            preview["combined_raw_weights"], preview["combined_weights"],
        ):
            self.assertAlmostEqual(raw, sample * loss)
            self.assertAlmostEqual(final, raw / preview["combined_mean"])
        self.assertAlmostEqual(sum(preview["combined_weights"]) / 5.0, 1.0)

    @patch("training_monitor_server.tr_train_base_counts", return_value=[489, 245, 245, 35, 70])
    def test_equalized_classes_can_legitimately_normalize_to_one(self, _counts):
        preview = automatic_tr_balance_preview(
            "split.json", "tracks.csv", [489] * 5,
            enabled=[True] * 5, sampling_enabled=[True] * 5, loss_enabled=[True] * 5,
            sampling_mode="inverse_frequency", sampling_boosts=[1.0] * 5,
            class_weight_mode="inverse_sqrt", beta=0.999, floor=0.25, cap=2.0,
        )
        self.assertTrue(all(abs(value - 1.0) < 1e-12 for value in preview["combined_weights"]))

    def test_creation_ui_places_method_above_training_and_diagnostic_augmentation(self):
        html = (ROOT / "scripts" / "training_monitor.html").read_text(encoding="utf-8")
        self.assertIn('data-suite-method-slot', html)
        self.assertIn('虚拟航迹方法同时作用于训练副本和启用的诊断副本', html)
        self.assertIn('样本数量已经包含在 P 中，不会重复相乘', html)

    def test_tr_clone_restores_training_and_nested_configuration_fields(self):
        html = (ROOT / "scripts" / "training_monitor.html").read_text(encoding="utf-8")
        self.assertIn("assign('batch_size',c.batch_size??c.batch_size_tr)", html)
        self.assertIn("c.split?.manifest", html)
        self.assertIn("c.skip_test??c.test_deferred??true", html)
        self.assertIn("partition==='train'?config.trajectory_augmentation?.targets:null", html)
        self.assertIn("payload.manual_sampling_boosts_enabled", html)

    def test_tr_ml_controls_sync_on_mode_plan_and_clone(self):
        html = (ROOT / "scripts" / "training_monitor.html").read_text(encoding="utf-8")
        self.assertIn("const syncMlControls=", html)
        self.assertIn("memberField.style.display=custom?'':'none'", html)
        self.assertIn("syncMlControls({resetRecommendedWeights:true})", html)
        self.assertIn("soft_lgbm_hist_svm:'2,2,1'", html)
        self.assertIn("const inferredPlan=", html)

    def test_rd_partition_expansion_uses_requested_targets(self):
        frames = [
            Frame(path="bird.mat", trajectory_id="1", source_target="BirdTarget", label=1),
            Frame(path="unknown.mat", trajectory_id="2", source_target="UnknownTarget", label=4),
        ]
        expanded, manifest = expand_rd_frames(
            frames, partition="train", targets=[1, 3, 1, 1, 2], seed=42,
        )
        self.assertEqual(manifest["expanded_counts"], [0, 3, 0, 0, 2])
        self.assertEqual(manifest["virtual_count"], 3)
        self.assertEqual(len(expanded), 5)
        self.assertTrue(all("|aug-train-" in frame.trajectory_id or frame.trajectory_id in {"1", "2"}
                            for frame in expanded))

    def test_partition_diagnostic_string_false_is_not_enabled(self):
        params = Handler.partition_augmentation_params({
            "partition_augmentation_diagnostics": "false",
            "partition_augmentation_targets_train": [1] * 5,
            "partition_augmentation_targets_val": [1] * 5,
            "partition_augmentation_targets_test": [1] * 5,
        })
        self.assertFalse(params["partition_augmentation_diagnostics"])
        self.assertEqual(params["partition_augmentation_method"], "perturbation")

    def test_partition_augmentation_method_is_validated(self):
        params = Handler.partition_augmentation_params({
            "partition_augmentation_method": "smote",
            "partition_augmentation_targets_train": [1] * 5,
            "partition_augmentation_targets_val": [1] * 5,
            "partition_augmentation_targets_test": [1] * 5,
        })
        self.assertEqual(params["partition_augmentation_method"], "smote")
        with self.assertRaises(ValueError):
            Handler.partition_augmentation_params({"partition_augmentation_method": "unknown"})

    def test_normalizes_all_three_experiment_types(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metric = {"accuracy": 0.8, "macro_f1": 0.7}
            rd = experiment_interface({"input_mode": "rd"}, {}, metric, {}, output)
            tr = experiment_interface({"experiment_type": "tr_only"}, {}, metric, {}, output)
            fusion = experiment_interface(
                {"experiment_type": "tr_rd_soft_cascade"},
                {"tr_branch": metric, "rd_branch": metric, "soft_cascade": metric},
                {}, {}, output,
            )
        self.assertEqual(rd["primary_branch"], "rd")
        self.assertEqual(tr["primary_branch"], "tr")
        self.assertEqual(fusion["primary_branch"], "fusion")
        self.assertEqual(set(fusion["branch_outputs"]), {"tr", "rd", "fusion"})

    def test_normalizes_running_tr_training_alias(self):
        normalized = experiment_interface(
            {"experiment_type": "tr_only_training"}, {}, {}, {}, ROOT
        )
        self.assertEqual(normalized["experiment_type"], "tr_only")

    def test_reads_branch_specific_saved_decisions(self):
        records = [{
            "trajectory_id": "42", "true_class": 1, "true_label": "bird",
            "tr_prediction": 1, "tr_prediction_label": "bird", "tr_probabilities": [0, 0.8, 0, 0, 0.2],
            "rd_prediction": 4, "rd_prediction_label": "other", "rd_probabilities": [0, 0.2, 0, 0, 0.8],
            "fused_prediction": 1, "fused_prediction_label": "bird", "fused_probabilities": [0, 0.9, 0, 0, 0.1],
            "rd_frame_count": 12, "rd_consistency": 0.75, "branch_agreement": False,
            "fusion_rescue_vs_tr": False, "fusion_harm_vs_tr": False,
        }]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "trajectory_decisions.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
            )
            tr_errors = read_trajectory_decisions(output, "tr", errors_only=True)
            rd_errors = read_trajectory_decisions(output, "rd", errors_only=True)
        self.assertEqual(tr_errors["total"], 0)
        self.assertEqual(rd_errors["total"], 1)
        self.assertEqual(rd_errors["items"][0]["prediction_label"], "other")
        self.assertEqual(rd_errors["items"][0]["rd_frame_count"], 12)

    def test_f_protocol_resume_selects_compatible_rd_cache(self):
        params = {
            "split_mode": "fixed_grouped", "grouped_split": str(DEFAULT_GROUPED_SPLIT),
            "max_train_frames_per_trajectory": 32, "target_width": 900,
            "velocity_min": -90.0, "velocity_max": 89.0, "resampling": "db_linear",
        }
        self.assertEqual(recommended_rd_cache(params), str(DEFAULT_F_SPLIT_RD_CACHE.resolve()))

    def test_cache_validation_rejects_non_contiguous_index(self):
        params = {
            "split_mode": "random", "velocity_min": -90.0, "velocity_max": 89.0,
            "target_width": 900, "resampling": "db_linear",
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "metadata.json").write_text(json.dumps({"preprocessing": {
                "velocity_min": -90.0, "velocity_max": 89.0,
                "target_width": 900, "resampling": "db_linear",
            }}), encoding="utf-8")
            (cache / "complete.json").write_text(json.dumps({"status": "complete", "frame_count": 2}), encoding="utf-8")
            (cache / "index.json").write_text(json.dumps({"a": 0, "b": 2}), encoding="utf-8")
            self.assertIn("index", validate_rd_cache(cache, params))


class SchedulerCommandTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = TrainingScheduler.__new__(TrainingScheduler)
        self.scheduler.root = ROOT

    def test_tr_and_fusion_commands_use_their_own_programs(self):
        tr_job = {
            "snapshot_path": str(ROOT / "scripts" / "evaluate_tr_only.py"),
            "output_dir": str(ROOT / "artifacts" / "tr_test"),
            "dataset_root": str(ROOT),
            "params": {
                "experiment_type": "tr_only", "grouped_split": "split.json",
                "track_index": "tracks.csv", "tr_checkpoint": "tr.pt",
                "partition": "val", "batch_size_tr": 32, "workers": 0,
            },
        }
        fusion_job = {
            "snapshot_path": str(ROOT / "scripts" / "evaluate_soft_cascade.py"),
            "output_dir": str(ROOT / "artifacts" / "fusion_test"),
            "dataset_root": "dataset", "params": {
                "experiment_type": "tr_rd_soft_cascade", "grouped_split": "split.json",
                "track_index": "tracks.csv", "tr_checkpoint": "tr.pt", "rd_checkpoint": "rd.pt",
                "partition": "val", "batch_size_tr": 32, "batch_size_rd": 128,
                "workers": 0, "fusion_mode": "fixed", "fixed_rd_weight": [0.2],
            },
        }
        tr_command = self.scheduler._command_for(tr_job)
        fusion_command = self.scheduler._command_for(fusion_job)
        self.assertIn("evaluate_tr_only.py", tr_command[1])
        self.assertIn("evaluate_soft_cascade.py", fusion_command[1])
        self.assertIn("--tr-checkpoint", fusion_command)
        self.assertIn("--rd-checkpoint", fusion_command)

    def test_tr_training_command_has_epoch_and_optimizer_contract(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "train_tr_only.py"),
            "output_dir": str(ROOT / "artifacts" / "tr_train_test"),
            "params": {
                "experiment_type": "tr_only", "grouped_split": "split.json",
                "track_index": "tracks.csv", "epochs": 24, "batch_size_tr": 8,
                "workers": 0, "learning_rate": 0.0002, "weight_decay": 0.0001,
                "patience": 0, "dropout": 0.1, "seed": 42, "skip_test": True,
            },
        }
        command = self.scheduler._command_for(job)
        self.assertIn("train_tr_only.py", command[1])
        self.assertIn("--epochs", command)
        self.assertIn("--learning-rate", command)
        self.assertIn("--dropout", command)
        self.assertIn("--bird-sample-boost", command)
        self.assertIn("--drone-sample-boost", command)
        self.assertIn("--balloon-sample-boost", command)
        self.assertIn("--partition-augmentation-targets-train", command)
        self.assertNotIn("--unknown-augmentation-copies", command)
        self.assertIn("--checkpoint-selection-metric", command)
        self.assertIn("--use-cosface", command)
        self.assertIn("--sampling-protocol", command)
        self.assertIn("--no-manual-sampling-boosts-enabled", command)
        self.assertNotIn("--checkpoint", command)

    def test_tr_ml_training_uses_ml_program_and_arguments(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "train_tr_ml.py"),
            "output_dir": str(ROOT / "artifacts" / "tr_ml_test"),
            "params": {
                "experiment_type": "tr_only", "tr_implementation": "ml",
                "grouped_split": "split.json", "track_index": "tracks.csv",
                "seed": 42, "ml_models": ["lightgbm", "rbf_svm"],
                "ml_model_weights": [2.0, 1.0], "ml_soft_voting": True,
                "skip_test": True,
            },
        }
        command = self.scheduler._command_for(job)
        self.assertIn("train_tr_ml.py", command[1])
        self.assertEqual(command[command.index("--models") + 1:command.index("--model-weights")], ["lightgbm", "rbf_svm"])
        self.assertIn("--soft-voting", command)
        self.assertIn("--skip-test", command)
        self.assertNotIn("--epochs", command)

    def test_tr_training_command_can_disable_cosface(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "train_tr_only.py"),
            "output_dir": str(ROOT / "artifacts" / "tr_no_cosface_test"),
            "params": {
                "experiment_type": "tr_only", "grouped_split": "split.json",
                "track_index": "tracks.csv", "use_cosface": False,
            },
        }
        command = self.scheduler._command_for(job)
        self.assertIn("--no-use-cosface", command)
        self.assertNotIn("--use-cosface", command)

    def test_tr_training_new_options_are_forwarded(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "train_tr_only.py"),
            "output_dir": str(ROOT / "artifacts" / "tr_options_test"),
            "params": {
                "experiment_type": "tr_only", "grouped_split": "split.json", "track_index": "tracks.csv",
                "lr_scheduler": "none", "bird_sample_boost": 1.3,
                "partition_augmentation_targets_train": [489, 489, 489, 489, 489],
                "partition_augmentation_method": "smote",
                "class_weight_mode": "class_balanced", "class_weight_floor": 0.25, "class_weight_cap": 2.0,
            },
        }
        command = self.scheduler._command_for(job)
        self.assertEqual(command[command.index("--lr-scheduler") + 1], "none")
        self.assertEqual(command[command.index("--bird-sample-boost") + 1], "1.3")
        self.assertEqual(command[command.index("--drone-sample-boost") + 1], "1.0")
        self.assertEqual(command[command.index("--balloon-sample-boost") + 1], "1.0")
        target_index = command.index("--partition-augmentation-targets-train")
        self.assertEqual(command[target_index + 1:target_index + 6], ["489"] * 5)
        self.assertEqual(command[command.index("--class-weight-floor") + 1], "0.25")
        self.assertEqual(command[command.index("--partition-augmentation-method") + 1], "smote")

    def test_tr_sampling_protocol_is_forwarded(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "train_tr_only.py"),
            "output_dir": str(ROOT / "artifacts" / "tr_sampling_protocol_test"),
            "dataset_root": str(ROOT),
            "params": {"experiment_type": "tr_only", "grouped_split": "split.json", "track_index": "tracks.csv",
                       "sampling_protocol": "strict_b01_replacement"},
        }
        command = self.scheduler._command_for(job)
        self.assertEqual(command[command.index("--sampling-protocol") + 1], "strict_b01_replacement")

    def test_tr_manual_sampling_mode_is_forwarded(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "train_tr_only.py"),
            "output_dir": str(ROOT / "artifacts" / "tr_manual_sampling_test"),
            "dataset_root": str(ROOT),
            "params": {"experiment_type": "tr_only", "grouped_split": "split.json", "track_index": "tracks.csv",
                       "manual_sampling_boosts_enabled": True},
        }
        command = self.scheduler._command_for(job)
        self.assertIn("--manual-sampling-boosts-enabled", command)
        self.assertNotIn("--no-manual-sampling-boosts-enabled", command)

    def test_tr_train_targets_are_retained_when_diagnostics_are_disabled(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "train_tr_only.py"),
            "output_dir": str(ROOT / "artifacts" / "tr_baseline_targets_test"),
            "params": {
                "experiment_type": "tr_only", "grouped_split": "split.json", "track_index": "tracks.csv",
                "partition_augmentation_diagnostics": False,
                "partition_augmentation_targets_train": [489, 245, 245, 35, 70],
                "partition_augmentation_targets_val": [105] * 5,
            },
        }
        command = self.scheduler._command_for(job)
        target_index = command.index("--partition-augmentation-targets-train")
        self.assertEqual(command[target_index + 1:target_index + 6], ["489", "245", "245", "35", "70"])
        self.assertNotIn("--partition-augmentation-targets-val", command)
        self.assertIn("--no-partition-augmentation-diagnostics", command)

    def test_gate_commands_do_not_receive_unsupported_partition_augmentation_flags(self):
        for experiment_type, program in (("fusion_gate_training", "generate_fusion_oof.py"),
                                         ("fusion_gate_calibration", "train_calibration_gate.py")):
            job = {
                "snapshot_path": str(ROOT / "scripts" / program),
                "output_dir": str(ROOT / "artifacts" / f"{experiment_type}_test"),
                "dataset_root": "dataset", "params": {
                    "experiment_type": experiment_type, "grouped_split": "split.json", "track_index": "tracks.csv",
                    "tr_checkpoint": "tr.pt", "rd_checkpoint": "rd.pt", "partition_augmentation_diagnostics": True,
                    "partition_augmentation_targets_train": [489] * 5,
                },
            }
            command = self.scheduler._command_for(job)
            self.assertNotIn("--partition-augmentation-diagnostics", command)
            self.assertNotIn("--partition-augmentation-targets-train", command)

    def test_checkpoint_evaluations_forward_partition_diagnostic_arguments(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "evaluate_rd_only.py"),
            "output_dir": str(ROOT / "artifacts" / "rd_eval_test"), "dataset_root": "dataset",
            "params": {"experiment_type": "rd_checkpoint_eval", "rd_checkpoint": "rd.pt", "partition": "val",
                       "partition_augmentation_diagnostics": True, "partition_augmentation_targets_val": [105] * 5},
        }
        command = self.scheduler._command_for(job)
        self.assertIn("--partition-augmentation-diagnostics", command)
        target_index = command.index("--partition-augmentation-targets-val")
        self.assertEqual(command[target_index + 1:target_index + 6], ["105"] * 5)

    def test_soft_cascade_metrics_is_a_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            self.assertTrue(self.scheduler._completion_exists(output))

    def test_soft_cascade_test_command_explicitly_allows_test(self):
        job = {
            "snapshot_path": str(ROOT / "scripts" / "evaluate_soft_cascade.py"),
            "output_dir": str(ROOT / "artifacts" / "fusion_test"),
            "dataset_root": "dataset", "params": {
                "experiment_type": "tr_rd_soft_cascade", "grouped_split": "split.json",
                "track_index": "tracks.csv", "tr_checkpoint": "tr.pt", "rd_checkpoint": "rd.pt",
                "partition": "test", "batch_size_tr": 32, "batch_size_rd": 128,
                "workers": 0, "fusion_mode": "fixed", "fixed_rd_weight": [0.2],
            },
        }
        self.assertIn("--allow-test", self.scheduler._command_for(job))

    def test_resume_in_place_keeps_same_job_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            output = artifacts / "run_a"
            output.mkdir(parents=True)
            checkpoint = output / "last.pt"
            checkpoint.write_bytes(b"checkpoint")
            scheduler = TrainingScheduler.__new__(TrainingScheduler)
            scheduler.root = root
            scheduler.artifacts = artifacts
            scheduler.lock = __import__("threading").RLock()
            scheduler.state_path = root / "scheduler.json"
            scheduler.state = {"jobs": [{
                "id": "job-a", "name": "run_a", "status": "failed", "order": 0,
                "output_dir": str(output), "params": {}, "attempts": 1,
            }]}
            scheduler._save_locked = lambda: None
            scheduler._program_for = lambda params: ("digest", root / "trainer.py")
            scheduler._command_for = lambda job: ["trainer", "--resume", job["params"]["resume_checkpoint"]]
            resumed = scheduler.resume_in_place("run_a", root, {"resume_checkpoint": str(checkpoint)})
            self.assertEqual(resumed["id"], "job-a")
            self.assertEqual(resumed["name"], "run_a")
            self.assertEqual(resumed["status"], "queued")
            self.assertEqual(resumed["resume_count"], 1)
            self.assertEqual(len(scheduler.state["jobs"]), 1)


if __name__ == "__main__":
    unittest.main()

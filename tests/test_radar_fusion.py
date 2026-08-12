from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from radar_fusion.model import SoftCascadeFusion, aggregate_rd_evidence
from radar_fusion.reporting import save_fusion_report
from radar_fusion.track_features import PHYSICAL_FEATURE_COLUMNS, TRACK_FEATURE_COLUMNS, encode_track
from radar_fusion.data import TrajectoryRecord, load_grouped_split
from radar_fusion.partition_augmentation import expand_rd_frames, expand_trajectory_records
from radar_fusion.trajectory_cache import load_or_build
from radar_fusion.trajectory_augmentation import append_training_copies, augment_track_frame, expand_training_records
from radar_rd.train import Frame


class FusionUnitTests(unittest.TestCase):
    @staticmethod
    def _write_track(path: Path, offset: float) -> None:
        length = 8
        base = np.linspace(0.0, 1.0, length)
        frame = pd.DataFrame({column: base + offset for column in TRACK_FEATURE_COLUMNS})
        frame["time_seconds"] = np.arange(length, dtype=float)
        frame["course_deg"] = (base * 20.0 + offset) % 360.0
        frame["nd_count"] = np.arange(length, dtype=float)
        frame.to_csv(path, index=False)

    def test_grouped_split_rejects_duplicate_partition_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({
                "train_group_ids": ["1"],
                "val_group_ids": ["1"],
                "test_group_ids": ["2"],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate trajectory assignments"):
                load_grouped_split(path, allow_subset=True)

    def test_rd_frame_probabilities_are_aggregated_per_track(self) -> None:
        frame_logits = torch.tensor([[4.0, 0, 0, 0, 0], [0, 4.0, 0, 0, 0], [0, 0, 4.0, 0, 0]])
        evidence = aggregate_rd_evidence(frame_logits, torch.tensor([0, 0, 1]), 2)
        self.assertEqual(evidence.frame_count.tolist(), [2.0, 1.0])
        self.assertEqual(evidence.predictions.tolist(), [0, 2])
        self.assertTrue(torch.all(evidence.available))

    def test_fixed_zero_weight_keeps_tr_prediction_and_records_rd(self) -> None:
        tr_logits = torch.tensor([[0.0, 5.0, 0, 0, 0], [0.0, 0, 0, 0, 5.0]])
        rd_logits = torch.tensor([[5.0, 0, 0, 0, 0], [0.0, 0, 0, 5.0, 0]])
        output = SoftCascadeFusion(mode="fixed", fixed_rd_weight=0.0)(
            tr_logits, rd_logits, torch.tensor([0, 1])
        )
        self.assertEqual(output.tr_predictions.tolist(), [1, 4])
        self.assertEqual(output.rd_predictions.tolist(), [0, 3])
        self.assertEqual(output.fused_predictions.tolist(), [1, 4])
        self.assertTrue(torch.allclose(output.rd_class_weights, torch.zeros_like(output.rd_class_weights)))

    def test_track_encoder_contract(self) -> None:
        frame = {column: [float(index), float(index + 1)] for index, column in enumerate(TRACK_FEATURE_COLUMNS)}
        frame["nd_count"] = [None, 2.0]
        import pandas as pd

        encoded = encode_track(pd.DataFrame(frame), height_missing=True, phase_missing=True)
        self.assertEqual(encoded.sequence.shape, (2, len(TRACK_FEATURE_COLUMNS)))
        self.assertEqual(encoded.physical.shape, (len(PHYSICAL_FEATURE_COLUMNS),))
        self.assertTrue(torch.tensor(encoded.sequence).isfinite().all())

    def test_reporting_retains_three_decisions_and_confusion_cases(self) -> None:
        records = [
            {
                "trajectory_id": "1",
                "true_class": 0,
                "tr_prediction": 0,
                "rd_prediction": 1,
                "fused_prediction": 0,
            },
            {
                "trajectory_id": "2",
                "true_class": 1,
                "tr_prediction": 1,
                "rd_prediction": 1,
                "fused_prediction": 2,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            summary = save_fusion_report(Path(temporary), records, {"partition": "val"})
            payload = json.loads((Path(temporary) / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["soft_cascade"]["confusion_cases"]["1:2"], ["2"])
            self.assertEqual(summary["complementarity"]["fusion_harm_vs_tr"], 1)

    def test_train_only_trajectory_augmentation_is_deterministic_and_target_is_extra_rows(self) -> None:
        records = [
            TrajectoryRecord(str(index), Path(f"track-{index}.csv"), 4, True, True)
            for index in range(2)
        ]
        expanded = append_training_copies(records, label=4, copies=3, kind="unknown_track_t1", seed=42)
        self.assertEqual(len(expanded), 5)
        self.assertEqual([item.trajectory_id for item in expanded[:2]], ["0", "1"])
        self.assertTrue(all(item.augmentation_kind == "unknown_track_t1" for item in expanded[2:]))
        frame = pd.DataFrame({column: list(range(8)) for column in TRACK_FEATURE_COLUMNS})
        a = augment_track_frame(frame, kind="unknown_track_t1", seed=123)
        b = augment_track_frame(frame, kind="unknown_track_t1", seed=123)
        self.assertTrue(a.equals(b))
        self.assertGreaterEqual(len(a), 3)
        self.assertEqual(len(frame), 8)

    def test_smote_uses_deterministic_same_class_feature_neighbours(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offsets = [0.0, 0.1, 0.2, 0.3, 0.4, 5.0, 20.0]
            records = []
            for index, offset in enumerate(offsets):
                path = root / f"track-{index}.csv"
                self._write_track(path, offset)
                records.append(TrajectoryRecord(str(index), path, 1, False, True))
            targets = [0, 8, 0, 0, 0]
            expanded_a, manifest_a = expand_trajectory_records(
                records, partition="train", targets=targets, seed=42, method="smote",
            )
            expanded_b, manifest_b = expand_trajectory_records(
                records, partition="train", targets=targets, seed=42, method="smote",
            )
            _, manifest_other_seed = expand_trajectory_records(
                records, partition="train", targets=targets, seed=43, method="smote",
            )
            virtual = manifest_a["records"][0]
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(manifest_a["expanded_counts"], [0, 8, 0, 0, 0])
            self.assertEqual(len(expanded_a), 8)
            self.assertNotEqual(virtual["source_trajectory_id"], virtual["source_trajectory_id_b"])
            self.assertIn(virtual["source_trajectory_id_b"], virtual["neighbor_candidates"])
            self.assertLessEqual(virtual["k_neighbors"], 5)
            self.assertNotEqual(
                virtual["interpolation_alpha"],
                manifest_other_seed["records"][0]["interpolation_alpha"],
            )

    def test_smote_neighbours_are_invariant_to_other_class_feature_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, offset in enumerate((0.0, 0.1, 0.2, 0.3, 0.4, 5.0, 20.0)):
                path = root / f"bird-{index}.csv"
                self._write_track(path, offset)
                records.append(TrajectoryRecord(f"bird-{index}", path, 1, False, True))
            targets = [0, 8, 0, 0, 0]
            _, reference = expand_trajectory_records(
                records, partition="train", targets=targets, seed=42, method="smote",
            )
            for index, offset in enumerate((-1e9, 1e9)):
                path = root / f"drone-{index}.csv"
                self._write_track(path, offset)
                records.append(TrajectoryRecord(f"drone-{index}", path, 0, False, True))
            targets_with_other_class = [2, 8, 0, 0, 0]
            _, with_other_class = expand_trajectory_records(
                records, partition="train", targets=targets_with_other_class, seed=42, method="smote",
            )
            self.assertEqual(reference["records"], with_other_class["records"])

    def test_smote_cache_and_fusion_rd_plan_are_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            rd_frames = []
            for index, offset in enumerate((0.0, 0.2, 0.4)):
                path = root / f"track-{index}.csv"
                self._write_track(path, offset)
                records.append(TrajectoryRecord(str(index), path, 4, False, True))
                rd_frames.append(Frame(str(root / f"rd-{index}.mat"), str(index), "Other", 4))
            targets = [0, 0, 0, 0, 4]
            expanded, tr_manifest = expand_trajectory_records(
                records, partition="val", targets=targets, seed=42, method="smote",
            )
            rd_expanded, rd_manifest = expand_rd_frames(
                rd_frames, partition="val", targets=targets, seed=42,
                method="smote", smote_plan=tr_manifest,
            )
            tr_virtual_ids = {item["trajectory_id"] for item in tr_manifest["records"]}
            rd_virtual_ids = {item["trajectory_id"] for item in rd_manifest["records"]}
            self.assertEqual(tr_virtual_ids, rd_virtual_ids)
            self.assertEqual({frame.trajectory_id for frame in rd_expanded} - {"0", "1", "2"}, tr_virtual_ids)

            cache_root = root / "cache"
            dataset_a, cache_a = load_or_build(expanded, cache_root)
            dataset_b, cache_b = load_or_build(expanded, cache_root)
            self.assertFalse(cache_a["hit"])
            self.assertTrue(cache_b["hit"])
            self.assertEqual(dataset_a[-1].sequence.shape[1], len(TRACK_FEATURE_COLUMNS))
            self.assertEqual(dataset_a[-1].physical.shape[0], len(PHYSICAL_FEATURE_COLUMNS))

    def test_trajectory_cache_context_invalidates_same_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = root / "track.csv"
            self._write_track(track, 0.0)
            records = [TrajectoryRecord("1", track, 1, False, True)]
            cache_root = root / "cache"
            _, first = load_or_build(
                records, cache_root,
                {"partition": "val", "method": "smote", "targets": [0, 2, 0, 0, 0], "seed": 42},
            )
            _, repeat = load_or_build(
                records, cache_root,
                {"partition": "val", "method": "smote", "targets": [0, 2, 0, 0, 0], "seed": 42},
            )
            _, changed_target = load_or_build(
                records, cache_root,
                {"partition": "val", "method": "smote", "targets": [0, 3, 0, 0, 0], "seed": 42},
            )
            self.assertFalse(first["hit"])
            self.assertTrue(repeat["hit"])
            self.assertFalse(changed_target["hit"])
            self.assertNotEqual(first["key"], changed_target["key"])


if __name__ == "__main__":
    unittest.main()

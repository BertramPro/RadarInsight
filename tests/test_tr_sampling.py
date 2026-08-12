import sys
import unittest
from types import SimpleNamespace

import torch

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from train_tr_only import (  # noqa: E402
    AuditedReplacementSampler,
    CoveragePlusBoostSampler,
    sampling_weights,
)


class TrajectorySamplingTests(unittest.TestCase):
    def setUp(self):
        self.records = [SimpleNamespace(label=value) for value in ([0] * 6 + [1] * 2 + [2] + [3])]

    def test_coverage_without_weighting_visits_every_record_once(self):
        weights, _ = sampling_weights(self.records, class_enabled=[False] * 5)
        sampler = CoveragePlusBoostSampler(self.records, weights, seed=42)
        sampler.set_epoch(1)
        indices = list(iter(sampler))
        self.assertEqual(len(indices), len(self.records))
        self.assertEqual(sorted(indices), list(range(len(self.records))))
        self.assertFalse(sampler.last_audit['duplicate_sampling'])
        self.assertEqual(sampler.last_audit['extra_count'], 0)

    def test_coverage_keeps_base_records_when_boosting(self):
        weights, _ = sampling_weights(
            self.records, sampling_mode='inverse_frequency',
            clutter_boost=5.0, class_enabled=[True] * 5,
        )
        sampler = CoveragePlusBoostSampler(self.records, weights, seed=42)
        sampler.set_epoch(2)
        indices = list(iter(sampler))
        self.assertLessEqual(len(indices) - len(self.records), len(self.records))
        self.assertEqual(set(range(len(self.records))), set(indices))
        self.assertGreaterEqual(sampler.last_audit['extra_class_counts'][3], 0)

    def test_coverage_is_deterministic_for_seed_and_epoch(self):
        weights, _ = sampling_weights(self.records, class_enabled=[True] * 5, clutter_boost=4.0)
        a = CoveragePlusBoostSampler(self.records, weights, seed=7)
        b = CoveragePlusBoostSampler(self.records, weights, seed=7)
        a.set_epoch(3); b.set_epoch(3)
        self.assertEqual(list(iter(a)), list(iter(b)))
        b.set_epoch(4)
        self.assertNotEqual(list(iter(a)), list(iter(b)))

    def test_strict_replacement_keeps_replacement_semantics(self):
        weights, _ = sampling_weights(self.records, class_enabled=[True] * 5)
        sampler = AuditedReplacementSampler(weights, num_samples=len(self.records), seed=1)
        indices = list(iter(sampler))
        self.assertEqual(len(indices), len(self.records))
        self.assertLessEqual(sampler.last_audit['unique_count'], len(self.records))
        self.assertEqual(sampler.last_audit['protocol'], 'strict_b01_replacement')


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from radar_fusion.ml_track import WeightedSoftVoting, clean_train_features, inverse_frequency_weights


class TestTRML(unittest.TestCase):
    def test_train_only_median_is_reused(self):
        train = np.array([[1.0, np.nan], [3.0, 8.0]])
        val = np.array([[np.nan, np.nan]])
        clean_train, clean_val, median = clean_train_features(train, val)
        np.testing.assert_allclose(median, [2.0, 8.0])
        np.testing.assert_allclose(clean_val, [[2.0, 8.0]])
        self.assertEqual(clean_train.shape[1], 2)

    def test_inverse_frequency_weights_are_mean_one(self):
        y = np.array([0, 0, 0, 1, 2, 3, 4])
        sample, classes = inverse_frequency_weights(y)
        self.assertAlmostEqual(float(sample.mean()), 1.0)
        self.assertGreater(classes[4], classes[0])

    def test_soft_voting_validates_members_and_weights(self):
        with self.assertRaises(ValueError):
            WeightedSoftVoting(["lightgbm"], [1.0], 42)
        with self.assertRaises(ValueError):
            WeightedSoftVoting(["lightgbm", "rbf_svm"], [1.0, 0.0], 42)


if __name__ == "__main__":
    unittest.main()

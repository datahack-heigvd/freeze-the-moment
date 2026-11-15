import unittest

import numpy as np
from kits.kit_vote_model import fill_vote_bins


class VoteFillTests(unittest.TestCase):
    def test_empty_support_keeps_fallback_available(self):
        original = np.zeros((4, 8, 3), np.float32)
        filled, support = fill_vote_bins(original)
        np.testing.assert_array_equal(filled, original)
        self.assertFalse(support.any())

    def test_full_support_unchanged(self):
        votes = np.arange(1, 49, dtype=np.float32).reshape(3, 8, 2)
        filled, support = fill_vote_bins(votes)
        np.testing.assert_array_equal(filled, votes)
        self.assertTrue(support.all())

    def test_single_supported_bin_fills_finitely(self):
        votes = np.zeros((40, 24, 2), np.float32)
        votes[20, 1] = [2, 3]
        filled, support = fill_vote_bins(votes)
        np.testing.assert_array_equal(filled, np.broadcast_to([2, 3], votes.shape))
        self.assertEqual(support.sum(), 1)

    def test_angular_seam_is_periodic(self):
        votes = np.zeros((1, 8, 2), np.float32)
        votes[0, 0] = [1, 0]
        votes[0, 4] = [0, 1]
        filled, _ = fill_vote_bins(votes)
        np.testing.assert_array_equal(filled[0, 7], [1, 0])

    def test_observed_bins_and_input_not_changed(self):
        votes = np.zeros((3, 8, 2), np.float32)
        votes[0, 1] = [1, 3]
        votes[2, 6] = [4, 1]
        old = votes.copy()
        filled, support = fill_vote_bins(votes)
        np.testing.assert_array_equal(filled[support], old[support])
        np.testing.assert_array_equal(votes, old)
        self.assertFalse(np.shares_memory(filled, votes))

    def test_arbitrarily_small_positive_support_is_not_dropped(self):
        votes = np.zeros((2, 4, 1), np.float32)
        votes[0, 0] = 1e-12
        filled, support = fill_vote_bins(votes)
        self.assertEqual(support.sum(), 1)
        self.assertTrue((filled > 0).all())

    def test_invalid_votes_rejected(self):
        for votes in (
            np.zeros((0, 8, 2)),
            np.zeros((2, 8)),
            np.full((2, 8, 1), -1),
            np.full((2, 8, 1), np.nan),
        ):
            with self.subTest(shape=votes.shape), self.assertRaises(ValueError):
                fill_vote_bins(votes)


if __name__ == "__main__":
    unittest.main()

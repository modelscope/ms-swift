import torch
import unittest

from swift.rl_core.advantage import get_local_rollout_values


class TestLocalRolloutValues(unittest.TestCase):

    def test_selects_each_ranks_original_values(self):
        values = torch.arange(10)
        sample_counts = [2, 3, 1, 4]

        local_values = [
            get_local_rollout_values(values, sample_counts, rollout_rank) for rollout_rank in range(len(sample_counts))
        ]

        torch.testing.assert_close(torch.cat(local_values), values)
        self.assertEqual([value.shape[0] for value in local_values], sample_counts)

    def test_rejects_values_from_a_different_sample_set(self):
        with self.assertRaisesRegex(AssertionError, 'Expected 4 rollout values'):
            get_local_rollout_values(torch.arange(8), [2, 2], rollout_rank=0)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import unittest

from swift.rl_core.advantage import compute_reward_metrics


class TestRewardMetrics(unittest.TestCase):

    @staticmethod
    def compute(rewards_per_func, names, num_generations=2, scale_rewards='group'):
        return compute_reward_metrics(
            rewards=rewards_per_func.nansum(dim=1),
            rewards_per_func=rewards_per_func,
            reward_func_names=names,
            num_generations=num_generations,
            scale_rewards=scale_rewards)

    def test_inapplicable_task_has_no_statistics(self):
        rewards = torch.tensor([[1., float('nan')], [3., float('nan')]])
        metrics = self.compute(rewards, ['present', 'absent'])
        self.assertEqual(metrics.per_func_mean, {'present': 2.0})
        self.assertEqual(set(metrics.per_func_std), {'present'})
        self.assertAlmostEqual(metrics.per_func_std['present'], math.sqrt(2), places=6)
        self.assertEqual(metrics.reward_mean, 2.0)
        self.assertAlmostEqual(metrics.reward_std, math.sqrt(2), places=6)

    def test_partial_task_coverage_uses_observed_rewards(self):
        rewards = torch.tensor([[1., float('nan')], [float('nan'), 4.], [3., float('nan')],
                                [float('nan'), float('nan')]])
        metrics = self.compute(rewards, ['two_samples', 'one_sample'])
        self.assertEqual(metrics.per_func_mean, {'two_samples': 2.0, 'one_sample': 4.0})
        self.assertAlmostEqual(metrics.per_func_std['two_samples'], math.sqrt(2), places=6)
        self.assertEqual(metrics.per_func_std['one_sample'], 0.0)

    def test_zero_rewards_are_observations(self):
        metrics = self.compute(torch.zeros(2, 1), ['zero'])
        self.assertEqual(metrics.per_func_mean, {'zero': 0.0})
        self.assertEqual(metrics.per_func_std, {'zero': 0.0})
        self.assertEqual(metrics.frac_reward_zero_std, 1.0)

    def test_absent_task_can_return_in_next_batch(self):
        absent = self.compute(torch.full((2, 1), float('nan')), ['task'])
        present = self.compute(torch.tensor([[2.], [4.]]), ['task'])
        self.assertNotIn('task', absent.per_func_mean)
        self.assertNotIn('task', absent.per_func_std)
        self.assertEqual(present.per_func_mean, {'task': 3.0})
        self.assertAlmostEqual(present.per_func_std['task'], math.sqrt(2), places=6)

    def test_single_generation_and_no_reward_functions(self):
        metrics = self.compute(torch.empty(1, 0), [], num_generations=1)
        self.assertEqual(metrics.per_func_mean, {})
        self.assertEqual(metrics.per_func_std, {})
        self.assertEqual(metrics.reward_mean, 0.0)
        self.assertEqual(metrics.reward_std, 0.0)


if __name__ == '__main__':
    unittest.main()

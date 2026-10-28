"""Driver counters and rollout budgets must agree with worker update boundaries."""
import unittest

from swift.dev.recipe.utils import _seed_counters, is_grad_sync_boundary, num_optimizer_steps, rollout_step_budget
from twinkle.model.optimizer_group import BaseOptimizerGroup


class AccumulationStepsTest(unittest.TestCase):
    def test_worker_driver_and_scheduler_count_complete_windows(self):
        for ga in (1, 2, 4, 8):
            group = BaseOptimizerGroup(gradient_accumulation_steps=ga)
            observed = 0
            for micro in range(0, 3 * ga + 1):
                group.cur_step = micro
                expected = micro > 0 and micro % ga == 0
                self.assertEqual(group.do_grad_sync(), expected)
                self.assertEqual(is_grad_sync_boundary(micro, ga), expected)
                observed += int(expected)
                self.assertEqual(num_optimizer_steps(micro, ga), observed)

    def test_three_grpo_batches_are_three_updates(self):
        # 24 prompts * 4 completions / 8 training ranks = 12 microbatches.
        self.assertEqual(rollout_step_budget(
            num_prompts=24, num_generations=4, train_batch_size=8,
            gradient_accumulation_steps=4, num_train_epochs=1,
            generation_batch_size=8), 3)
        self.assertEqual(num_optimizer_steps(4, 4), 1)
        self.assertEqual(num_optimizer_steps(3, 4), 0)

    def test_resume_from_complete_window(self):
        self.assertEqual(_seed_counters({'cur_step': 12, 'consumed_train_samples': 96},
                                       gradient_accumulation_steps=4, dataloader=None,
                                       ignore_data_skip=False), (3, 12, 0))


if __name__ == '__main__':
    unittest.main()

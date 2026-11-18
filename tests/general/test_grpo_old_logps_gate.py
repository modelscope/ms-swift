# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from types import SimpleNamespace

from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

_need_old_per_token_logps = GRPOTrainer._need_old_per_token_logps


def make_trainer(**kwargs):
    trainer = dict(
        old_policy=lambda: False,
        kl_in_reward=False,
        beta=0.0,
        args=SimpleNamespace(async_generate=False, use_liger_kernel=False),
    )
    trainer.update(kwargs)
    return SimpleNamespace(**trainer)


def make_batch(**kwargs):
    batch = dict(teacher_per_token_logps=None, rollout_per_token_logps=None)
    batch.update(kwargs)
    return SimpleNamespace(**batch)


class TestNeedOldPerTokenLogps(unittest.TestCase):
    """The extra forward over the pre-update policy is only skipped when it is provably redundant."""

    def test_default_setup_skips_the_forward(self):
        # num_iterations=1, single optimizer step, no reader without a fallback.
        self.assertFalse(_need_old_per_token_logps(make_trainer(), make_batch()))

    def test_multiple_iterations_keep_the_forward(self):
        self.assertTrue(_need_old_per_token_logps(make_trainer(old_policy=lambda: True), make_batch()))

    def test_kl_in_reward_keeps_the_forward_only_with_beta(self):
        # kl_in_reward computes its KL straight from the field, with no `per_token_logps` fallback.
        self.assertTrue(_need_old_per_token_logps(make_trainer(kl_in_reward=True, beta=0.1), make_batch()))
        self.assertFalse(_need_old_per_token_logps(make_trainer(kl_in_reward=True, beta=0.0), make_batch()))

    def test_teacher_kl_keeps_the_forward(self):
        batch = make_batch(teacher_per_token_logps=object())
        self.assertTrue(_need_old_per_token_logps(make_trainer(), batch))

    def test_rollout_logps_keep_the_forward(self):
        # The Liger path reads the field directly to build the rollout importance ratio.
        batch = make_batch(rollout_per_token_logps=object())
        self.assertTrue(_need_old_per_token_logps(make_trainer(), batch))

    def test_async_generate_keeps_the_forward(self):
        trainer = make_trainer(args=SimpleNamespace(async_generate=True))
        self.assertTrue(_need_old_per_token_logps(trainer, make_batch()))

    def test_liger_kernel_keeps_the_forward(self):
        trainer = make_trainer(args=SimpleNamespace(async_generate=False, use_liger_kernel=True))
        self.assertTrue(_need_old_per_token_logps(trainer, make_batch()))


if __name__ == '__main__':
    unittest.main()

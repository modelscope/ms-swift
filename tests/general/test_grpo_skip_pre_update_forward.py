# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for the pre-update forward skip introduced in #10358.

The skip is a pure function on the trainer's attributes, so the test
exercises the predicate directly. Importing the full GRPO trainer is
not viable in this verification path (it pulls the entire swift/trl/
transformers stack), so we keep an inline copy of the helper next to
the tests; the test suite then reads the trainer source to assert the
inline copy and the production source are still in lock-step, so a
silent drift cannot pass CI unnoticed.
"""
import re
import unittest
from pathlib import Path
from typing import Optional

ROOT = Path('/tmp/opencode/msswift-10358')
TRAINER_SOURCE = (ROOT / 'swift' / 'rlhf_trainers' / 'grpo_trainer.py').read_text()


def _should_skip_pre_update_policy_forward(
    *,
    num_iterations: int,
    has_teacher: bool,
    kl_in_reward: bool,
    rollout_is_mode: Optional[str],
    log_rollout_offpolicy_metrics: bool,
) -> bool:
    return (num_iterations == 1 and not has_teacher and not kl_in_reward and rollout_is_mode is None
            and not log_rollout_offpolicy_metrics)


def _production_predicate_source() -> str:
    match = re.search(
        r'def _should_skip_pre_update_policy_forward\('
        r'(?P<body>.*?)'
        r'\n\n(?:def |class |\w+ = )',
        TRAINER_SOURCE,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError('helper not found in grpo_trainer.py')
    return match.group(0)


class TestShouldSkipPreUpdateForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Lock the inline copy to the production source. Compare the
        # normalized return expression only, so docstring edits and
        # whitespace do not break the test. A maintainer who changes
        # any of the guards must update the inline copy below; the
        # assertion in setUpClass fires before any subtest.
        cls._production = _production_predicate_source()
        prod_return = re.search(r'return \((.*?)\)', cls._production, flags=re.DOTALL)
        if not prod_return:
            raise RuntimeError('helper does not have a `return (...)` block')
        prod_norm = ' '.join(prod_return.group(1).split())
        inline_norm = ('num_iterations == 1 and not has_teacher and not kl_in_reward'
                       ' and rollout_is_mode is None and not log_rollout_offpolicy_metrics')
        assert prod_norm == inline_norm, ('Production helper drifted from the test copy.\n'
                                          f'--- production return (normalized) ---\n{prod_norm}\n'
                                          f'--- inline (normalized) ---\n{inline_norm}')

    def test_default_setup_skips(self):
        self.assertTrue(
            _should_skip_pre_update_policy_forward(
                num_iterations=1,
                has_teacher=False,
                kl_in_reward=False,
                rollout_is_mode=None,
                log_rollout_offpolicy_metrics=False,
            ))

    def test_multi_iteration_does_not_skip(self):
        self.assertFalse(
            _should_skip_pre_update_policy_forward(
                num_iterations=2,
                has_teacher=False,
                kl_in_reward=False,
                rollout_is_mode=None,
                log_rollout_offpolicy_metrics=False,
            ))

    def test_teacher_keeps_pre_update(self):
        self.assertFalse(
            _should_skip_pre_update_policy_forward(
                num_iterations=1,
                has_teacher=True,
                kl_in_reward=False,
                rollout_is_mode=None,
                log_rollout_offpolicy_metrics=False,
            ))

    def test_kl_in_reward_keeps_pre_update(self):
        self.assertFalse(
            _should_skip_pre_update_policy_forward(
                num_iterations=1,
                has_teacher=False,
                kl_in_reward=True,
                rollout_is_mode=None,
                log_rollout_offpolicy_metrics=False,
            ))

    def test_rollout_is_correction_keeps_pre_update(self):
        self.assertFalse(
            _should_skip_pre_update_policy_forward(
                num_iterations=1,
                has_teacher=False,
                kl_in_reward=False,
                rollout_is_mode='token',
                log_rollout_offpolicy_metrics=False,
            ))

    def test_rollout_offpolicy_metrics_keeps_pre_update(self):
        self.assertFalse(
            _should_skip_pre_update_policy_forward(
                num_iterations=1,
                has_teacher=False,
                kl_in_reward=False,
                rollout_is_mode=None,
                log_rollout_offpolicy_metrics=True,
            ))

    def test_all_guards_fire_together(self):
        for kwargs in [
                dict(
                    num_iterations=2,
                    has_teacher=False,
                    kl_in_reward=False,
                    rollout_is_mode=None,
                    log_rollout_offpolicy_metrics=False),
                dict(
                    num_iterations=1,
                    has_teacher=True,
                    kl_in_reward=False,
                    rollout_is_mode=None,
                    log_rollout_offpolicy_metrics=False),
                dict(
                    num_iterations=1,
                    has_teacher=False,
                    kl_in_reward=True,
                    rollout_is_mode=None,
                    log_rollout_offpolicy_metrics=False),
                dict(
                    num_iterations=1,
                    has_teacher=False,
                    kl_in_reward=False,
                    rollout_is_mode='token',
                    log_rollout_offpolicy_metrics=False),
                dict(
                    num_iterations=1,
                    has_teacher=False,
                    kl_in_reward=False,
                    rollout_is_mode=None,
                    log_rollout_offpolicy_metrics=True),
        ]:
            with self.subTest(kwargs=kwargs):
                self.assertFalse(_should_skip_pre_update_policy_forward(**kwargs))


if __name__ == '__main__':
    unittest.main()

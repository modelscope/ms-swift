import unittest
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

from swift.rlhf_trainers.args_mixin import RolloutTrainerArgumentsMixin
from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin


class TestRolloutOffloadOrder(unittest.TestCase):

    def _make_trainer(self, events, rollout_error=None, reset_error=None):
        trainer = object.__new__(RolloutTrainerMixin)
        args = object.__new__(RolloutTrainerArgumentsMixin)
        args.sleep_level = 1
        trainer.args = args
        trainer.vllm_mode = 'colocate'
        trainer.enable_offload = True
        trainer.async_generate = False
        trainer.state = SimpleNamespace(global_step=0)
        trainer._last_loaded_step = 0
        trainer.request_config = Mock()

        def reset_prefix_cache():
            events.append('reset_vllm_cache')
            if reset_error is not None:
                raise reset_error

        trainer.engine = SimpleNamespace(
            inner_model_executor=SimpleNamespace(is_sleeping=False),
            engine=SimpleNamespace(
                reset_prefix_cache=reset_prefix_cache,
                sleep=lambda level: events.append(f'sleep_vllm_{level}'),
            ),
        )

        @contextmanager
        def offload_context(_self):
            events.append('offload_trainer')
            try:
                yield
            finally:
                events.append('load_trainer')

        def infer(_self, samples, request_config):
            events.append('rollout')
            if rollout_error is not None:
                raise rollout_error
            return samples

        trainer.offload_context = MethodType(offload_context, trainer)
        trainer._infer_single_or_multi_turn = MethodType(infer, trainer)
        trainer.multi_turn_completion_length_context = lambda: nullcontext()
        return trainer

    @patch('swift.rlhf_trainers.utils.set_expandable_segments')
    @patch('swift.rlhf_trainers.utils.aggressive_empty_cache')
    def test_vllm_sleeps_before_trainer_reload(self, _, __):
        events = []
        trainer = self._make_trainer(events)

        result = trainer._fast_infer(['sample'])

        self.assertEqual(result, ['sample'])
        self.assertEqual(events, [
            'offload_trainer',
            'rollout',
            'reset_vllm_cache',
            'sleep_vllm_1',
            'load_trainer',
        ])

    @patch('swift.rlhf_trainers.utils.set_expandable_segments')
    @patch('swift.rlhf_trainers.utils.aggressive_empty_cache')
    def test_vllm_sleeps_before_trainer_reload_when_rollout_fails(self, _, __):
        events = []
        rollout_error = RuntimeError('rollout failed')
        trainer = self._make_trainer(events, rollout_error)

        with self.assertRaisesRegex(RuntimeError, 'rollout failed'):
            trainer._fast_infer(['sample'])

        self.assertEqual(events, [
            'offload_trainer',
            'rollout',
            'reset_vllm_cache',
            'sleep_vllm_1',
            'load_trainer',
        ])

    @patch('swift.rlhf_trainers.utils.set_expandable_segments')
    @patch('swift.rlhf_trainers.utils.aggressive_empty_cache')
    def test_cleanup_error_does_not_mask_rollout_error(self, _, __):
        events = []
        rollout_error = RuntimeError('rollout failed')
        reset_error = RuntimeError('cache reset failed')
        trainer = self._make_trainer(events, rollout_error, reset_error)

        with self.assertRaisesRegex(RuntimeError, 'rollout failed'):
            trainer._fast_infer(['sample'])

        self.assertEqual(events, [
            'offload_trainer',
            'rollout',
            'reset_vllm_cache',
            'sleep_vllm_1',
            'load_trainer',
        ])


if __name__ == '__main__':
    unittest.main()

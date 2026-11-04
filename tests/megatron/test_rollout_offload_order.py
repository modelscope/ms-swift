import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import patch


class TestMegatronRolloutOffloadOrder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from swift.megatron.trainers import rollout_mixin
            cls.rollout_mixin = rollout_mixin
            cls.MegatronRolloutMixin = rollout_mixin.MegatronRolloutMixin
            cls._megatron_available = True
        except (ImportError, RuntimeError) as error:
            cls._megatron_available = False
            cls._skip_reason = str(error)

    def _skip_if_no_megatron(self):
        if not self._megatron_available:
            self.skipTest(f'Megatron dependencies not available: {self._skip_reason}')

    def _make_trainer(self, events, *, step, last_loaded_step, wake_up_supports_tags=True):
        self._skip_if_no_megatron()
        trainer = object.__new__(self.MegatronRolloutMixin)
        trainer.args = SimpleNamespace(sleep_level=1, max_turns=None, report_to=[])
        trainer.vllm_mode = 'colocate'
        trainer.enable_offload = True
        trainer._step = step
        trainer._last_loaded_step = last_loaded_step
        trainer.multi_turn_scheduler = None
        trainer.enable_server_multi_turn = False

        inner_model_executor = SimpleNamespace(is_sleeping=True)

        def reset_prefix_cache():
            events.append('reset_vllm_cache')

        def wake_up_with_tags(tags=None):
            events.append(f'wake_vllm_{"+".join(tags) if tags else "all"}')
            if not tags or 'kv_cache' in tags:
                inner_model_executor.is_sleeping = False

        def wake_up_without_tags():
            events.append('wake_vllm_all')
            inner_model_executor.is_sleeping = False

        wake_up = wake_up_with_tags if wake_up_supports_tags else wake_up_without_tags

        trainer.engine = SimpleNamespace(
            inner_model_executor=inner_model_executor,
            engine=SimpleNamespace(
                reset_prefix_cache=reset_prefix_cache,
                wake_up=wake_up,
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

        trainer.offload_context = MethodType(offload_context, trainer)
        trainer._preprocess_inputs = MethodType(lambda _self, samples: samples, trainer)
        trainer._move_model_to_vllm = MethodType(lambda _self: events.append('sync_weights_to_vllm'), trainer)
        trainer._rollout = MethodType(lambda _self, samples: events.append('rollout') or samples, trainer)
        trainer._postprocess_rollout_outputs = MethodType(lambda _self, samples, _outputs: samples, trainer)
        return trainer

    def test_same_step_offloads_before_waking_sleeping_vllm(self):
        events = []
        trainer = self._make_trainer(events, step=1, last_loaded_step=1)

        with patch.object(self.rollout_mixin, 'aggressive_empty_cache'), \
                patch.object(self.rollout_mixin, 'set_expandable_segments'):
            result = trainer._generate_completions(['sample'])

        self.assertEqual(result, ['sample'])
        self.assertEqual(events, [
            'offload_trainer',
            'wake_vllm_weights+kv_cache',
            'rollout',
            'reset_vllm_cache',
            'sleep_vllm_1',
            'load_trainer',
        ])

    def test_new_step_syncs_weights_before_offloading_trainer(self):
        events = []
        trainer = self._make_trainer(events, step=1, last_loaded_step=0)

        with patch.object(self.rollout_mixin, 'aggressive_empty_cache'), \
                patch.object(self.rollout_mixin, 'set_expandable_segments'):
            result = trainer._generate_completions(['sample'])

        self.assertEqual(result, ['sample'])
        self.assertEqual(events, [
            'wake_vllm_weights',
            'sync_weights_to_vllm',
            'offload_trainer',
            'wake_vllm_kv_cache',
            'rollout',
            'reset_vllm_cache',
            'sleep_vllm_1',
            'load_trainer',
        ])

    def test_legacy_vllm_offloads_before_waking_all_memory(self):
        events = []
        trainer = self._make_trainer(events, step=1, last_loaded_step=1, wake_up_supports_tags=False)

        with patch.object(self.rollout_mixin, 'aggressive_empty_cache'), \
                patch.object(self.rollout_mixin, 'set_expandable_segments'):
            result = trainer._generate_completions(['sample'])

        self.assertEqual(result, ['sample'])
        self.assertEqual(events, [
            'offload_trainer',
            'wake_vllm_all',
            'rollout',
            'reset_vllm_cache',
            'sleep_vllm_1',
            'load_trainer',
        ])


if __name__ == '__main__':
    unittest.main()

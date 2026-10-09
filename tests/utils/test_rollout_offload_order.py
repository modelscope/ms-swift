import torch
import unittest
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

from swift.rlhf_trainers.args_mixin import RolloutTrainerArgumentsMixin
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin


class TestRolloutOffloadOrder(unittest.TestCase):

    def _make_trainer(self, events, rollout_error=None, reset_error=None, is_sleeping=False, global_step=0):
        trainer = object.__new__(RolloutTrainerMixin)
        args = object.__new__(RolloutTrainerArgumentsMixin)
        args.sleep_level = 1
        trainer.args = args
        trainer.vllm_mode = 'colocate'
        trainer.enable_offload = True
        trainer.async_generate = False
        trainer.state = SimpleNamespace(global_step=global_step)
        trainer._last_loaded_step = 0
        trainer.request_config = Mock()

        inner_model_executor = SimpleNamespace(is_sleeping=is_sleeping)

        def reset_prefix_cache():
            events.append('reset_vllm_cache')
            if reset_error is not None:
                raise reset_error

        def wake_up(tags=None):
            events.append(f'wake_vllm_{"+".join(tags) if tags else "all"}')
            if not tags or 'kv_cache' in tags:
                inner_model_executor.is_sleeping = False

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

        def infer(_self, samples, request_config):
            events.append('rollout')
            if rollout_error is not None:
                raise rollout_error
            return samples

        trainer.offload_context = MethodType(offload_context, trainer)
        trainer._infer_single_or_multi_turn = MethodType(infer, trainer)
        trainer.multi_turn_completion_length_context = lambda: nullcontext()
        trainer._move_model_to_vllm = MethodType(lambda _self, **_kwargs: events.append('sync_weights_to_vllm'),
                                                 trainer)
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

    @patch('swift.rlhf_trainers.utils.set_expandable_segments')
    @patch('swift.rlhf_trainers.utils.aggressive_empty_cache')
    def test_same_step_offloads_before_waking_sleeping_vllm(self, _, __):
        events = []
        trainer = self._make_trainer(events, is_sleeping=True)

        result = trainer._fast_infer(['sample'])

        self.assertEqual(result, ['sample'])
        self.assertEqual(events, [
            'offload_trainer',
            'wake_vllm_weights+kv_cache',
            'rollout',
            'reset_vllm_cache',
            'sleep_vllm_1',
            'load_trainer',
        ])

    @patch('swift.rlhf_trainers.utils.set_expandable_segments')
    @patch('swift.rlhf_trainers.utils.aggressive_empty_cache')
    def test_new_step_syncs_weights_before_offloading_trainer(self, _, __):
        events = []
        trainer = self._make_trainer(events, is_sleeping=True, global_step=1)

        result = trainer._fast_infer(['sample'])

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


class TestSelectiveModelReload(unittest.TestCase):

    def _make_trainer(self, offload_results, ref_model=None):
        trainer = object.__new__(GRPOTrainer)
        actor_model = Mock(name='actor_model')
        wrapped_model = Mock(name='wrapped_model')
        trainer.args = SimpleNamespace(offload_model=True, offload_optimizer=False)
        trainer.model = wrapped_model
        trainer.ref_model = ref_model
        trainer.accelerator = SimpleNamespace(unwrap_model=Mock(return_value=actor_model))
        trainer.offload_model = Mock(side_effect=offload_results)
        trainer.load_model = Mock()
        trainer.optimizer = None
        return trainer, actor_model

    def test_cpu_resident_actor_is_not_reloaded(self):
        trainer, actor_model = self._make_trainer([False])

        with trainer.offload_context():
            pass

        trainer.offload_model.assert_called_once_with(actor_model)
        trainer.load_model.assert_not_called()

    def test_only_models_actually_offloaded_are_reloaded(self):
        ref_model = Mock(name='ref_model')
        trainer, actor_model = self._make_trainer([False, True], ref_model)

        with trainer.offload_context():
            pass

        self.assertEqual(trainer.offload_model.call_args_list, [
            unittest.mock.call(actor_model),
            unittest.mock.call(ref_model),
        ])
        trainer.load_model.assert_called_once_with(ref_model)

    def test_gpu_resident_actor_is_reloaded(self):
        trainer, actor_model = self._make_trainer([True])

        with trainer.offload_context():
            pass

        trainer.load_model.assert_called_once_with(actor_model)

    @patch('swift.rlhf_trainers.rollout_mixin.is_deepspeed_enabled', return_value=False)
    @patch('swift.rlhf_trainers.rollout_mixin.torch.cuda.empty_cache')
    def test_cpu_model_reports_that_nothing_moved(self, empty_cache, _):
        trainer = object.__new__(RolloutTrainerMixin)
        trainer._is_fsdp2 = False
        model = torch.nn.Linear(2, 2)

        moved = trainer.offload_model(model)

        self.assertFalse(moved)
        empty_cache.assert_not_called()
        self.assertTrue(all(param.device.type == 'cpu' for param in model.parameters()))


if __name__ == '__main__':
    unittest.main()

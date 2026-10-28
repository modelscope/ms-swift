# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import tempfile
import types
import unittest
from functools import partial
from unittest import mock

from swift.utils import LAST_CHECKPOINT_SYMLINK, update_last_checkpoint_symlink

try:
    import swift.megatron.utils.megatron_lm_utils as megatron_lm_utils
    MEGATRON_UNAVAILABLE = None
except Exception as e:
    # CI does not install megatron, and a mismatched mcore dependency can fail with anything from
    # ImportError to AttributeError, so any failure here means the tests below cannot run.
    megatron_lm_utils = None
    MEGATRON_UNAVAILABLE = f'Megatron dependencies not available: {e}'


class FakeAsyncRequest:

    def __init__(self):
        self.finalize_fns = []

    def add_finalize_fn(self, fn):
        self.finalize_fns.append(fn)


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestAsyncStrategyFallback(unittest.TestCase):
    """mcore defaults its async save to `nvrx`, which is unusable without a recent nvidia-resiliency-ext."""

    def test_no_kwargs_for_synchronous_save(self):
        self.assertEqual(megatron_lm_utils._get_async_strategy_kwargs(False), {})

    def test_falls_back_to_mcore_without_nvrx(self):
        with mock.patch('megatron.core.dist_checkpointing.strategies.torch.HAVE_NVRX', False):
            self.assertEqual(megatron_lm_utils._get_async_strategy_kwargs(True), {'async_strategy': 'mcore'})

    def test_keeps_the_default_with_nvrx(self):
        with mock.patch('megatron.core.dist_checkpointing.strategies.torch.HAVE_NVRX', True):
            self.assertEqual(megatron_lm_utils._get_async_strategy_kwargs(True), {})


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestAsyncSaveSymlink(unittest.TestCase):
    """Under `--async_save` the weights are still being written when `save_mcore_checkpoint` returns."""

    def _save(self, async_save: bool):
        M = megatron_lm_utils
        root = tempfile.mkdtemp()
        checkpoint_dir = os.path.join(root, 'checkpoint-2')
        os.makedirs(checkpoint_dir)
        request = FakeAsyncRequest()
        args = types.SimpleNamespace(output_dir=root, async_save=async_save, data_parallel_random_init=False)

        dist_checkpointing = mock.MagicMock()
        dist_checkpointing.save.return_value = request if async_save else None
        with mock.patch.object(M, 'unwrap_model', lambda model: model), \
             mock.patch.object(M, '_get_rng_state', dict), \
             mock.patch.object(M, 'get_sharded_sd_metadata', lambda args: {}), \
             mock.patch.object(M, '_generate_state_dict', lambda *args, **kwargs: {'weight': 1}), \
             mock.patch.object(M, '_filter_adapter_state_dict', lambda state_dict, peft_format: None), \
             mock.patch.object(M, 'mcore_017', True), \
             mock.patch.object(M, 'TorchDistSaveShardedStrategy', object, create=True), \
             mock.patch.object(M, 'FullyParallelSaveStrategyWrapper', lambda strategy, group: strategy), \
             mock.patch.object(M, 'mpu', mock.MagicMock()), \
             mock.patch.object(M, 'dist_checkpointing', dist_checkpointing), \
             mock.patch.object(M, 'is_master', lambda: True), \
             mock.patch.object(M, 'schedule_async_save', lambda request: None), \
             mock.patch.object(M.torch.distributed, 'is_initialized', lambda: False):
            deferred = M.save_mcore_checkpoint(
                args, [object()],
                iteration=2,
                output_dir=checkpoint_dir,
                async_finalize_fn=partial(update_last_checkpoint_symlink, checkpoint_dir))
        return deferred, request, os.path.join(root, LAST_CHECKPOINT_SYMLINK)

    def test_symlink_waits_for_the_asynchronous_save(self):
        deferred, request, link_path = self._save(True)
        self.assertTrue(deferred)
        # The symlink update is deferred to the finalize callbacks instead of running right away.
        self.assertEqual(len(request.finalize_fns), 2)
        self.assertFalse(os.path.lexists(link_path))
        for fn in request.finalize_fns:
            fn()
        self.assertEqual(os.readlink(link_path), 'checkpoint-2')

    def test_synchronous_save_leaves_the_symlink_to_the_caller(self):
        deferred, request, link_path = self._save(False)
        self.assertFalse(deferred)
        self.assertEqual(request.finalize_fns, [])
        self.assertFalse(os.path.lexists(link_path))


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestFsdpExportPreparation(unittest.TestCase):

    def _save(self, root, *, fsdp=True, export=True, no_save_optim=False, merge=False, skip_adapter=False):
        import swift.megatron.trainers.base as trainer_module

        args = types.SimpleNamespace(
            output_dir=root,
            use_megatron_fsdp=fsdp,
            save_safetensors=export,
            no_save_optim=no_save_optim,
            tuner_type='lora' if merge or skip_adapter else 'full',
            merge_lora=merge,
            save_missing_weights=False)
        wrapper = mock.Mock()
        bare_model = mock.Mock()
        bridge = mock.Mock(spec=['save_weights'])
        events = mock.Mock()
        events.attach_mock(wrapper.start_param_sync, 'gather')
        events.attach_mock(wrapper.module._replace_param_with_raw_if_needed, 'restore')
        events.attach_mock(bridge.save_weights, 'export')
        trainer = types.SimpleNamespace(
            args=args,
            state=types.SimpleNamespace(consumed_train_samples=0, iteration=2, best_global_step=None),
            wrapped_models=[wrapper],
            unwrapped_models=[bare_model],
            optimizer=None,
            opt_param_scheduler=None,
            bridge=bridge,
            template=types.SimpleNamespace(processor=None),
            copy_path=mock.Mock(),
            merge_lora_adapters=events.merge,
            unmerge_lora_adapters=events.unmerge)
        if skip_adapter:
            args.tuner_type = 'lora_llm'
        save = mock.Mock(return_value=False)
        events.attach_mock(save, 'checkpoint')
        with mock.patch.object(trainer_module, 'save_mcore_checkpoint', save), \
             mock.patch.object(trainer_module, 'gc_collect'), \
             mock.patch.object(trainer_module, 'is_master', return_value=False):
            trainer_module.BaseMegatronTrainer.save_checkpoint(trainer)
        self.assertEqual(save.call_args.args[1], [] if export and no_save_optim else [wrapper])
        bare_model.start_param_sync.assert_not_called()
        return events

    def test_export_preparation_is_independent_of_dcp_saving(self):
        for fsdp in (False, True):
            for export in (False, True):
                for no_save_optim in (False, True):
                    with self.subTest(fsdp=fsdp, export=export, no_save_optim=no_save_optim), \
                         tempfile.TemporaryDirectory() as root:
                        events = self._save(root, fsdp=fsdp, export=export, no_save_optim=no_save_optim)
                        expected = ['checkpoint']
                        if fsdp and export:
                            expected += ['gather', 'restore']
                            events.gather.assert_called_once_with(force_sync=True)
                            events.restore.assert_called_once_with()
                        if export:
                            expected += ['export']
                        self.assertEqual([call[0] for call in events.mock_calls], expected)

    def test_merge_prepares_parameters_when_adapter_export_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            events = self._save(root, no_save_optim=True, merge=True)
            self.assertEqual([call[0] for call in events.mock_calls],
                             ['checkpoint', 'gather', 'restore', 'merge', 'export', 'unmerge'])

    def test_no_preparation_when_both_export_and_merge_are_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            events = self._save(root, skip_adapter=True)
            self.assertEqual([call[0] for call in events.mock_calls], ['checkpoint'])


if __name__ == '__main__':
    unittest.main()

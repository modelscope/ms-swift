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
        args = types.SimpleNamespace(
            output_dir=root, async_save=async_save, no_save_rng=False, data_parallel_random_init=False)

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


if __name__ == '__main__':
    unittest.main()

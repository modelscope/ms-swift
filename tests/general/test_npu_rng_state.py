# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from transformers.trainer import ParallelMode
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swift.trainers import mixin


class TestNpuRngState(unittest.TestCase):
    """The NPU RNG state has to be checkpointed, otherwise a resumed run diverges from an uninterrupted one."""

    @staticmethod
    def _rng_states(parallel_mode, cuda_available=False):
        trainer = SimpleNamespace(args=SimpleNamespace(parallel_mode=parallel_mode))
        with patch.object(mixin, 'is_torch_npu_available', return_value=True), \
                patch.object(mixin, 'is_torch_musa_available', return_value=False), \
                patch.object(mixin.torch.cuda, 'is_available', return_value=cuda_available):
            return mixin.SwiftMixin._build_rng_states(trainer)

    def test_saves_the_npu_rng_state(self):
        npu = Mock()
        npu.random.get_rng_state.return_value = 'npu-rng-state'
        with patch.object(mixin.torch, 'npu', npu, create=True):
            rng_states = self._rng_states(ParallelMode.NOT_DISTRIBUTED)

        self.assertEqual(rng_states['npu'], 'npu-rng-state')
        npu.random.get_rng_state.assert_called_once_with()

    def test_saves_every_rank_of_the_npu_rng_state_when_distributed(self):
        npu = Mock()
        npu.random.get_rng_state_all.return_value = ['rank0-rng-state', 'rank1-rng-state']
        with patch.object(mixin.torch, 'npu', npu, create=True):
            rng_states = self._rng_states(ParallelMode.DISTRIBUTED)

        self.assertEqual(rng_states['npu'], ['rank0-rng-state', 'rank1-rng-state'])
        npu.random.get_rng_state_all.assert_called_once_with()

    def test_omits_the_npu_key_without_torch_npu(self):
        trainer = SimpleNamespace(args=SimpleNamespace(parallel_mode=ParallelMode.NOT_DISTRIBUTED))
        with patch.object(mixin, 'is_torch_npu_available', return_value=False), \
                patch.object(mixin, 'is_torch_musa_available', return_value=False), \
                patch.object(mixin.torch.cuda, 'is_available', return_value=False):
            rng_states = mixin.SwiftMixin._build_rng_states(trainer)

        self.assertEqual(set(rng_states), {'python', 'numpy', 'cpu'})


if __name__ == '__main__':
    unittest.main()

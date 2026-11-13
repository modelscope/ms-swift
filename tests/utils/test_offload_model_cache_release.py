# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin


class _StubParameter:

    def __init__(self, device_type):
        self.device = SimpleNamespace(type=device_type)
        self.data = torch.zeros(2)


class _StubModule:
    """A model stand-in that reports its weights on `device_type`.

    `offload_model` only reads `parameters()`, `buffers()` and `device.type`, so a stub
    keeps the test independent of an installed accelerator.
    """

    def __init__(self, device_type):
        self._device_type = device_type
        self._parameters = [_StubParameter(device_type)]
        self.cpu_called = False

    def parameters(self):
        return iter(self._parameters)

    def buffers(self):
        return iter([])

    def cpu(self):
        self.cpu_called = True
        self._parameters = [_StubParameter('cpu')]
        return self


class TestOffloadModelCacheRelease(unittest.TestCase):
    """`offload_model` hands the freed device memory back to the active accelerator.

    The method moves the weights to the CPU and then releases the device cache so that the
    freed memory is available to the rollout engine.  Releasing through
    `torch.cuda.empty_cache()` does nothing on an Ascend NPU (or any other non-CUDA
    accelerator), so the device memory stayed in the caching allocator; these tests pin the
    release to the shared `swift.utils.torch_utils.empty_cache` helper, which dispatches to
    NPU / MUSA / MPS / CUDA.
    """

    def _make_mixin(self, is_fsdp2):
        mixin = object.__new__(RolloutTrainerMixin)
        mixin._is_fsdp2 = is_fsdp2
        return mixin

    @patch('swift.rlhf_trainers.rollout_mixin.empty_cache')
    def test_fsdp2_offload_releases_the_accelerator_cache(self, empty_cache):
        module = _StubModule('npu')

        self.assertTrue(self._make_mixin(is_fsdp2=True).offload_model(module))

        self.assertTrue(module.cpu_called)
        empty_cache.assert_called_once_with()

    @patch('swift.rlhf_trainers.rollout_mixin.empty_cache')
    def test_offload_releases_the_accelerator_cache(self, empty_cache):
        module = _StubModule('npu')

        self.assertTrue(self._make_mixin(is_fsdp2=False).offload_model(module))

        empty_cache.assert_called_once_with()

    @patch('swift.rlhf_trainers.rollout_mixin.empty_cache')
    def test_offload_without_movement_keeps_the_cache(self, empty_cache):
        module = _StubModule('cpu')

        self.assertFalse(self._make_mixin(is_fsdp2=True).offload_model(module))

        empty_cache.assert_not_called()

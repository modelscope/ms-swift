# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-Ascend memory compatibility patches used by NPU GRPO rollout.

The patches here are limited to memory accounting and profiling behavior.  They
do not change model math.  Their job is to keep vLLM's KV-cache capacity
estimation stable when vLLM is colocated with a training process that may
release or reclaim NPU memory during profiling.
"""
from __future__ import annotations

import torch
from contextlib import contextmanager


def _patch_vllm_ascend_mem_get_info() -> None:
    """Make ``NPUPlatform.mem_get_info`` tolerant of torch-npu signatures."""
    try:
        from vllm_ascend.platform import NPUPlatform
    except (ImportError, AttributeError):
        return
    if getattr(NPUPlatform, '_swift_mem_get_info_patched', False):
        return

    @classmethod
    def mem_get_info(cls, device=None):
        if device is None:
            return torch.npu.mem_get_info()
        try:
            return torch.npu.mem_get_info(device=device)
        except TypeError:
            return torch.npu.mem_get_info()

    NPUPlatform.mem_get_info = mem_get_info
    NPUPlatform._swift_mem_get_info_patched = True


def _patch_vllm_ascend_memory_profiling() -> None:
    """Normalize vLLM-Ascend profile results when free memory increases.

    In colocated GRPO, another rank or the training side can release memory
    while vLLM is profiling.  Upstream profiling assumes non-vLLM memory is
    stable and can assert if free memory goes up.  The wrapper keeps the profile
    result internally consistent instead of failing the rollout initialization.
    """
    try:
        from vllm.logger import logger as vllm_logger
        from vllm_ascend.worker import worker as ascend_worker
    except (ImportError, AttributeError, RuntimeError):
        return

    origin_memory_profiling = getattr(ascend_worker, 'memory_profiling', None)
    if origin_memory_profiling is None or getattr(origin_memory_profiling, '_swift_ascend_memory_profiling_patched',
                                                  False):
        return

    @contextmanager
    def memory_profiling(baseline_snapshot, weights_memory: int = 0):
        with origin_memory_profiling(baseline_snapshot, weights_memory=weights_memory) as profile_result:
            yield profile_result

        if profile_result.after_profile.free_memory < baseline_snapshot.free_memory:
            return

        # In colocated GRPO, the training side may release NPU memory while
        # vLLM-Ascend is profiling KV cache capacity. vLLM-Ascend assumes
        # external memory is stable and asserts when free memory increases.
        profile_result.after_profile.free_memory = max(baseline_snapshot.free_memory - 1, 0)
        profile_result.after_profile.cuda_memory = (
            profile_result.after_profile.total_memory - profile_result.after_profile.free_memory)
        profile_result.non_torch_increase = max(profile_result.non_torch_increase, 0)
        profile_result.torch_peak_increase = max(profile_result.torch_peak_increase, 0)
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase + profile_result.torch_peak_increase + profile_result.weights_memory)
        vllm_logger.warning_once('Patched vLLM-Ascend memory profiling because free memory increased during profiling. '
                                 'This is expected in colocated training when another rank releases memory.')

    memory_profiling._swift_ascend_memory_profiling_patched = True
    memory_profiling._swift_origin = origin_memory_profiling
    ascend_worker.memory_profiling = memory_profiling


def patch_vllm_ascend_memory_runtime() -> None:
    """Apply non-colocate-specific vLLM-Ascend memory runtime patches."""
    _patch_vllm_ascend_mem_get_info()
    _patch_vllm_ascend_memory_profiling()


def _patch_vllm_ascend_colocate_memory_profiling() -> None:
    """Patch ``NPUWorker.determine_available_memory`` for colocated GRPO.

    vLLM-Ascend normally treats increased free memory during profiling as an
    error.  In colocated training that increase is expected when the training
    side frees memory.  This wrapper keeps the profiled non-KV memory result and
    continues with a conservative available KV-cache estimate.
    """
    try:
        from vllm.utils.mem_constants import GiB_bytes
        from vllm.utils.mem_utils import memory_profiling
        from vllm_ascend.worker import worker as ascend_worker
    except (ImportError, AttributeError, RuntimeError):
        return

    NPUWorker = getattr(ascend_worker, 'NPUWorker', None)
    if NPUWorker is None:
        return
    origin_determine = getattr(NPUWorker, 'determine_available_memory', None)
    if origin_determine is None or getattr(origin_determine, '_swift_colocate_memory_patched', False):
        return

    @torch.inference_mode()
    def determine_available_memory(self) -> int:

        def GiB(num_bytes):
            return num_bytes / GiB_bytes

        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        if self.init_snapshot.free_memory <= free_gpu_memory:
            ascend_worker.logger.warning(
                'vLLM-Ascend memory profiling observed increased free memory '
                'during colocate initialization: initial %.2f GiB, current %.2f GiB. '
                'Continuing with profiled non-KV memory instead of failing.', GiB(self.init_snapshot.free_memory),
                GiB(free_gpu_memory))

        self.available_kv_cache_memory_bytes = self.requested_memory - profile_result.non_kv_cache_memory
        ascend_worker.logger.debug(profile_result)
        ascend_worker.logger.info_once(
            'Available KV cache memory: %.2f GiB', GiB(self.available_kv_cache_memory_bytes), scope='local')
        return int(self.available_kv_cache_memory_bytes)

    determine_available_memory._swift_colocate_memory_patched = True
    determine_available_memory._swift_origin = origin_determine
    NPUWorker.determine_available_memory = determine_available_memory


def patch_vllm_ascend_colocate_runtime() -> None:
    """Apply vLLM-Ascend memory patches needed by colocated training."""
    _patch_vllm_ascend_colocate_memory_profiling()


__all__ = [
    'patch_vllm_ascend_colocate_runtime',
    'patch_vllm_ascend_memory_runtime',
]

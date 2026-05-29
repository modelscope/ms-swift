# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM-Ascend memory compatibility patches used by NPU GRPO rollout."""
from __future__ import annotations

import torch


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


def patch_vllm_ascend_memory_runtime() -> None:
    """Apply non-colocate-specific vLLM-Ascend memory runtime patches."""
    _patch_vllm_ascend_mem_get_info()


__all__ = [
    'patch_vllm_ascend_memory_runtime',
]

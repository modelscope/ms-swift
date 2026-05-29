# Copyright (c) ModelScope Contributors. All rights reserved.
"""Small vLLM-Ascend memory API compatibility patches.

This module intentionally avoids colocate memory-policy changes.  It only
normalizes API differences that are safe for both standalone vLLM-Ascend
inference and SWIFT GRPO rollout.
"""
from __future__ import annotations

import torch


def _patch_vllm_ascend_mem_get_info() -> None:
    """Patch ``NPUPlatform.mem_get_info`` for torch-npu binding differences.

    vLLM-Ascend calls ``current_platform.mem_get_info(device)`` during worker
    initialization.  Without this wrapper, some versions expose
    ``NPUPlatform.mem_get_info`` in a way that gets Python method binding plus
    the explicit device argument at the same time, producing:

        TypeError: mem_get_info() got multiple values for argument 'device'

    Defining a classmethod here gives vLLM-Ascend one stable call surface.  It
    keeps the device-aware torch-npu query when available and falls back to the
    no-argument query only when torch-npu rejects the keyword.  This does not
    change memory profiling policy.
    """
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
    """Apply memory patches that do not depend on colocated training."""
    _patch_vllm_ascend_mem_get_info()


__all__ = [
    'patch_vllm_ascend_memory_runtime',
]

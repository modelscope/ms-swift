# Copyright (c) ModelScope Contributors. All rights reserved.
"""Top-level vLLM-Ascend runtime patch entrypoints for SWIFT NPU support.

This module is intentionally a facade.  The actual compatibility code lives in
smaller modules grouped by responsibility: process-group lifecycle, memory
compatibility, and MoE execution/weight sync.  Existing callers keep importing from this
file, while maintainers can review each patch family independently.
"""
from __future__ import annotations

import sys

from swift.model.npu_patch.vllm_ascend_groups import (get_or_create_vllm_tp_gloo_group,
                                                      patch_vllm_ascend_external_launcher_groups,
                                                      prepare_vllm_ascend_device_groups_before_megatron,
                                                      prepare_vllm_ascend_dp_groups_before_megatron,
                                                      register_megatron_hccl_groups_for_vllm)
from swift.model.npu_patch.vllm_ascend_memory import patch_vllm_ascend_memory_runtime
from swift.model.npu_patch.vllm_ascend_moe import (patch_vllm_ascend_moe_expert_weight_loader,
                                                   patch_vllm_ascend_moe_runtime)
from swift.utils.logger import get_logger

logger = get_logger()


def _patch_flash_attn_optional_import() -> None:
    """Clear a stub ``flash_attn`` module that can block optional imports.

    Some stacks insert a non-package ``flash_attn`` placeholder into
    ``sys.modules``.  vLLM import paths then treat it as the real package and
    fail on submodule imports.  Removing the placeholder lets normal optional
    dependency checks proceed.
    """
    module = sys.modules.get('flash_attn')
    if module is None or hasattr(module, '__path__'):
        return
    for module_name in list(sys.modules):
        if module_name == 'flash_attn' or module_name.startswith('flash_attn.'):
            sys.modules.pop(module_name, None)


def patch_vllm_ascend_runtime(*, colocate: bool = False) -> None:
    """Apply SWIFT runtime compatibility patches for vLLM-Ascend on NPU.

    The caller is responsible for guarding this function by device type.  This
    module intentionally imports vLLM/vLLM-Ascend modules lazily so CUDA/GPU
    paths do not enter Ascend-only code.  Colocated training process-group
    patches are applied only when the caller marks the runtime as colocated.
    """
    _patch_flash_attn_optional_import()
    patch_vllm_ascend_moe_runtime()
    patch_vllm_ascend_memory_runtime()
    if colocate:
        patch_vllm_ascend_external_launcher_groups()


__all__ = [
    'get_or_create_vllm_tp_gloo_group',
    'patch_vllm_ascend_moe_expert_weight_loader',
    'patch_vllm_ascend_runtime',
    'prepare_vllm_ascend_device_groups_before_megatron',
    'prepare_vllm_ascend_dp_groups_before_megatron',
    'register_megatron_hccl_groups_for_vllm',
]

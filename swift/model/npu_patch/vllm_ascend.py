# Copyright (c) ModelScope Contributors. All rights reserved.
"""Facade for SWIFT's vLLM-Ascend NPU compatibility patches.

Keep this file thin.  The real patches are split by responsibility:

* ``vllm_ascend_moe``: MoE routing and GRPO weight-sync layout handling.
* ``vllm_ascend_memory``: small torch-npu/vLLM-Ascend memory API compatibility.
* ``vllm_ascend_groups``: colocated Megatron + vLLM process-group lifecycle.

Callers should import from this module so the public entrypoints stay stable,
while reviewers can audit each patch family in its own file.  The caller is
still responsible for guarding these entrypoints with an NPU/device check.
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
    """Apply vLLM-Ascend patches needed by SWIFT NPU rollout.

    ``colocate=False`` covers patches that are also safe for standalone
    vLLM-Ascend server/native inference, such as optional import cleanup, MoE
    routing, and ``mem_get_info`` binding compatibility.

    ``colocate=True`` additionally patches process-group creation for Megatron
    GRPO colocate mode, where vLLM runs inside an already-initialized Megatron
    distributed process tree.
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

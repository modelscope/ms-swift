# Copyright (c) ModelScope Contributors. All rights reserved.
"""Public facade for vLLM-Ascend group helpers used by SWIFT NPU patches."""
from __future__ import annotations

from swift.model.npu_patch.vllm_ascend_group_control import get_or_create_vllm_tp_gloo_group
from swift.model.npu_patch.vllm_ascend_group_factory import (prepare_vllm_ascend_device_groups_before_megatron,
                                                             prepare_vllm_ascend_dp_groups_before_megatron)
from swift.model.npu_patch.vllm_ascend_group_registry import (create_npu_process_group,
                                                              register_megatron_hccl_groups_for_vllm)
from swift.model.npu_patch.vllm_ascend_group_runtime import patch_vllm_ascend_external_launcher_groups

__all__ = [
    'create_npu_process_group',
    'get_or_create_vllm_tp_gloo_group',
    'patch_vllm_ascend_external_launcher_groups',
    'prepare_vllm_ascend_device_groups_before_megatron',
    'prepare_vllm_ascend_dp_groups_before_megatron',
    'register_megatron_hccl_groups_for_vllm',
]

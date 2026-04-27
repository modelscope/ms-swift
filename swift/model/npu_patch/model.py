# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import torch
import torch_npu
from torch import nn
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

from swift.utils.logger import get_logger

from .moe import NpuMoeFused, npu_moe_block_forward
from .utils import apply_patch_map, import_optional_module

logger = get_logger()


class NpuRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'


class NpuQwen3_5RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        scale = (1.0 + self.weight).to(dtype=x.dtype)
        return torch_npu.npu_rms_norm(x, scale, epsilon=self.eps)[0]

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.eps}'


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def npu_apply_rotary_pos_emb_qwen3_5(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = torch_npu.npu_rotary_mul(q_rot, cos, sin)
    k_rot = torch_npu.npu_rotary_mul(k_rot, cos, sin)

    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


def npu_swiglu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1))


def _patch_transformers_flash_linear_attention_available() -> None:

    def _is_flash_linear_attention_available() -> bool:
        return True

    transformers_utils = import_optional_module('transformers.utils')
    if transformers_utils is not None:
        setattr(transformers_utils, 'is_flash_linear_attention_available', _is_flash_linear_attention_available)

    transformers_import_utils = import_optional_module('transformers.utils.import_utils')
    if transformers_import_utils is not None:
        setattr(transformers_import_utils, 'is_flash_linear_attention_available', _is_flash_linear_attention_available)


def patch_qwen3_5_chunk_gated_delta_rule_with_mindspeed() -> None:
    try:
        from ..chunk_gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        logger.warning('Failed to import embedded MindSpeed chunk_gated_delta_rule: %s', exc)
        return

    patched_modules = []
    for module_name in ('transformers.models.qwen3_5.modeling_qwen3_5',
                        'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe'):
        module = import_optional_module(module_name)
        if module is None:
            continue

        setattr(module, 'is_flash_linear_attention_available', lambda: True)
        setattr(module, 'is_fast_path_available', True)
        # FLA's fused RMSNormGated initializes with torch.cuda.current_device(),
        # so keep the native Qwen3.5 torch implementation on NPU.
        setattr(module, 'FusedRMSNormGated', None)
        setattr(module, 'chunk_gated_delta_rule', chunk_gated_delta_rule)
        patched_modules.append(module_name)

    if patched_modules:
        logger.info('Patched Qwen3.5 chunk_gated_delta_rule to embedded MindSpeed implementation: %s.',
                    ', '.join(patched_modules))


def _get_patch_table(modeling_qwen3_5):
    return (
        (
            modeling_qwen2,
            {
                'Qwen2RMSNorm': NpuRMSNorm,
                'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
                'Qwen2MLP.forward': npu_swiglu_forward,
            },
        ),
        (
            modeling_qwen3,
            {
                'Qwen3RMSNorm': NpuRMSNorm,
                'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
                'Qwen3MLP.forward': npu_swiglu_forward,
            },
        ),
        *(((
            modeling_qwen3_5,
            {
                'Qwen3_5RMSNorm': NpuQwen3_5RMSNorm,
                'apply_rotary_pos_emb': npu_apply_rotary_pos_emb_qwen3_5,
                'Qwen3_5MLP.forward': npu_swiglu_forward,
            },
        ), ) if modeling_qwen3_5 is not None else ()),
        (
            modeling_qwen3_moe,
            {
                'Qwen3MoeRMSNorm': NpuRMSNorm,
                'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
                'Qwen3MoeSparseMoeBlock.forward': npu_moe_block_forward,
            },
        ),
        (
            modeling_qwen3_vl_moe,
            {
                'Qwen3VLMoeTextExperts.forward': NpuMoeFused.npu_moe_experts_forward,
                'Qwen3VLMoeTextSparseMoeBlock.forward': NpuMoeFused.npu_moe_sparse_block_forward,
                'Qwen3VLMoeTextRMSNorm': NpuRMSNorm,
                'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
            },
        ),
    )


_APPLIED = False


def apply_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return

    modeling_qwen3_5 = import_optional_module('transformers.models.qwen3_5.modeling_qwen3_5')
    if modeling_qwen3_5 is not None:
        _patch_transformers_flash_linear_attention_available()
        patch_qwen3_5_chunk_gated_delta_rule_with_mindspeed()

    for module, patch_map in _get_patch_table(modeling_qwen3_5):
        apply_patch_map(module, patch_map)

    _APPLIED = True

# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

from swift.utils.logger import get_logger
from .utils import apply_patch_map, import_optional_module

logger = get_logger()

# ---------------------------------------------------------------------------
# Common NPU helpers
# ---------------------------------------------------------------------------


def _resolve_unsqueeze_dim(position_ids=None, unsqueeze_dim=1):
    if isinstance(position_ids, int) and unsqueeze_dim == 1:
        return position_ids
    return unsqueeze_dim


def _get_hidden_size(module, hidden_states: torch.Tensor) -> int:
    return getattr(module, 'hidden_size', getattr(module, 'hidden_dim', hidden_states.shape[-1]))


class NpuRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'


class NpuGmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, group_list, split_size):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        ctx.split_size = split_size

        outputs = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, group_type=0, split_item=2)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        wt = weight.permute(0, 2, 1)
        xt = x.permute(1, 0)
        dx = torch_npu.npu_grouped_matmul([grad_outputs], [wt], group_list=group_list, group_type=0, split_item=2)
        split_size = ctx.split_size
        xt_list = torch.split(xt, split_size, dim=1)
        grad_outputs_list = torch.split(grad_outputs, split_size, dim=0)
        with torch.npu.amp.autocast(enabled=False):
            dw = torch.stack([torch.matmul(xt_list[i], grad_outputs_list[i]) for i in range(len(xt_list))])

        return dx[0], dw, None, None


class GmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        fwd_output = torch_npu.npu_grouped_matmul([x], [weight],
                                                  bias=None,
                                                  group_list=group_list,
                                                  split_item=2,
                                                  group_type=0,
                                                  group_list_type=1)[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list

        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul([grad_output], [weight],
                                                  bias=None,
                                                  group_list=group_list,
                                                  split_item=2,
                                                  group_type=0,
                                                  group_list_type=1)[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_input, grad_weight, None


def _normalize_packed_expert_weights(module, input_dtype: torch.dtype, hidden_dim: int):
    gate_up_proj = module.gate_up_proj.to(input_dtype)
    down_proj = module.down_proj.to(input_dtype)

    if gate_up_proj.shape[1] == hidden_dim:
        gate_up_weight = gate_up_proj
    elif gate_up_proj.shape[2] == hidden_dim:
        gate_up_weight = gate_up_proj.transpose(1, 2)
    else:
        raise RuntimeError(f'Unsupported gate_up_proj shape for NPU MoE patch: {tuple(gate_up_proj.shape)}.')

    if down_proj.shape[2] == hidden_dim:
        down_weight = down_proj
    elif down_proj.shape[1] == hidden_dim:
        down_weight = down_proj.transpose(1, 2)
    else:
        raise RuntimeError(f'Unsupported down_proj shape for NPU MoE patch: {tuple(down_proj.shape)}.')

    return gate_up_weight, down_weight


def npu_packed_moe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    router_indices_or_routing_weights: torch.Tensor,
    routing_weights_or_router_indices: torch.Tensor,
) -> torch.Tensor:
    if router_indices_or_routing_weights.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        router_indices = router_indices_or_routing_weights
        routing_weights = routing_weights_or_router_indices
    else:
        routing_weights = router_indices_or_routing_weights
        router_indices = routing_weights_or_router_indices

    output_shape = hidden_states.shape
    hidden_dim = output_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)

    if routing_weights.shape != router_indices.shape:
        routing_weights = torch.gather(routing_weights, dim=-1, index=router_indices.to(torch.long))
    routing_weights = routing_weights.to(hidden_states.dtype)
    router_indices = router_indices.to(torch.int32)

    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, router_indices)
    tokens_per_expert = torch.histc(
        router_indices.to(torch.float), bins=self.num_experts, min=0, max=self.num_experts).to(torch.int64)
    gate_up_weight, down_weight = _normalize_packed_expert_weights(self, hidden_states.dtype, hidden_dim)

    intermediate_hidden_states = GmmFunction.apply(permuted_hidden_states, gate_up_weight, tokens_per_expert)
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    output = GmmFunction.apply(intermediate_activations, down_weight, tokens_per_expert)
    next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
    return next_states.view(*output_shape)


def _topk_from_router_logits(module, hidden_states: torch.Tensor, router_logits: torch.Tensor):
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, router_indices = torch.topk(routing_weights, module.top_k, dim=-1)
    if getattr(module, 'norm_topk_prob', True):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    return routing_weights, router_indices


# ---------------------------------------------------------------------------
# Qwen2/Qwen3 dense patch
# ---------------------------------------------------------------------------


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    unsqueeze_dim = _resolve_unsqueeze_dim(position_ids, unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def npu_swiglu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1))


QWEN2_PATCHES = {
    'Qwen2RMSNorm': NpuRMSNorm,
    'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
    'Qwen2MLP.forward': npu_swiglu_forward,
}

QWEN3_PATCHES = {
    'Qwen3RMSNorm': NpuRMSNorm,
    'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
    'Qwen3MLP.forward': npu_swiglu_forward,
}

# ---------------------------------------------------------------------------
# Qwen3.5 dense patch
# ---------------------------------------------------------------------------


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


def npu_apply_rotary_pos_emb_qwen3_5(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    unsqueeze_dim = _resolve_unsqueeze_dim(position_ids, unsqueeze_dim)
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


QWEN3_5_PATCHES = {
    'Qwen3_5RMSNorm': NpuQwen3_5RMSNorm,
    'apply_rotary_pos_emb': npu_apply_rotary_pos_emb_qwen3_5,
    'Qwen3_5MLP.forward': npu_swiglu_forward,
}

# ---------------------------------------------------------------------------
# Qwen3-MoE patch
# ---------------------------------------------------------------------------


def _qwen3_moe_forward_transformers_457(self, hidden_states: torch.Tensor,
                                        router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if getattr(self, 'norm_topk_prob', False):
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    input_dtype = hidden_states.dtype
    up_weight_list = [expert.up_proj.weight.t().to(input_dtype) for expert in self.experts]
    gate_weight_list = [expert.gate_proj.weight.t().to(input_dtype) for expert in self.experts]
    down_weight_list = [expert.down_proj.weight.t().to(input_dtype) for expert in self.experts]
    w1 = torch.stack(up_weight_list)
    w2 = torch.stack(gate_weight_list)
    w3 = torch.stack(down_weight_list)

    routing_map = selected_experts
    flatten_indices = routing_map.view(-1)
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    permuted_tokens = hidden_states.index_select(0, sorted_indices // self.top_k)

    tokens_per_experts = torch.sum(expert_mask, dim=(1, 2))
    group_list = torch.cumsum(tokens_per_experts, dim=0)

    cpu_group_list = group_list.to('cpu', non_blocking=False)
    cpu_group_list = [0] + cpu_group_list.tolist()
    split_size = [cpu_group_list[i + 1] - cpu_group_list[i] for i in range(len(cpu_group_list) - 1)]

    up_res = NpuGmmFunction.apply(permuted_tokens, w1, group_list, split_size)
    gate_res = NpuGmmFunction.apply(permuted_tokens, w2, group_list, split_size)
    act_res = torch_npu.npu_swiglu(torch.cat([gate_res, up_res], dim=-1))
    down_res = NpuGmmFunction.apply(act_res, w3, group_list, split_size)

    num_unpermuted_tokens = routing_weights.numel()
    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, down_res.shape[-1]],
        dtype=down_res.dtype,
        device=down_res.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, down_res)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, self.top_k, down_res.size(-1))
    unpermuted_tokens = unpermuted_tokens * routing_weights.unsqueeze(-1)
    final_hidden_states = unpermuted_tokens.sum(dim=1).to(hidden_states.dtype)
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    return final_hidden_states, router_logits


def _qwen3_moe_forward_transformers_5(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor,
                                      selected_experts: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    final_hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def npu_qwen3_moe_sparse_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_dim = hidden_states.shape[-1]
    gate_output = self.gate(hidden_states.view(-1, hidden_dim))

    if isinstance(gate_output, tuple):
        # Transformers 5.x: gate is a router module and returns
        # (router_logits, routing_weights, selected_experts).
        _, routing_weights, selected_experts = gate_output
        return _qwen3_moe_forward_transformers_5(self, hidden_states, routing_weights, selected_experts)

    # Transformers 4.57.x: gate is nn.Linear and returns router logits.
    return _qwen3_moe_forward_transformers_457(self, hidden_states, gate_output)


QWEN3_MOE_PATCHES = {
    'Qwen3MoeRMSNorm': NpuRMSNorm,
    'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
    'Qwen3MoeSparseMoeBlock.forward': npu_qwen3_moe_sparse_block_forward,
}

QWEN3_MOE_TRANSFORMERS_5_PATCHES = {
    'Qwen3MoeExperts.forward': npu_packed_moe_experts_forward,
}

# ---------------------------------------------------------------------------
# Qwen3-VL-MoE patch
# ---------------------------------------------------------------------------


def _qwen3_vl_moe_forward_transformers_457(self, hidden_states: torch.Tensor,
                                           router_logits: torch.Tensor) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    hidden_size = _get_hidden_size(self, hidden_states)
    hidden_states = hidden_states.reshape(-1, hidden_size)

    routing_weights, router_indices = _topk_from_router_logits(self, hidden_states, router_logits)
    hidden_states = hidden_states.reshape(batch_size, -1, hidden_size)
    return self.experts(hidden_states, routing_weights, router_indices)


def _qwen3_vl_moe_forward_transformers_5(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor,
                                         selected_experts: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_size)
    routed_out = self.experts(hidden_states, selected_experts, routing_weights)
    return routed_out.reshape(batch_size, sequence_length, hidden_size)


def npu_qwen3_vl_moe_sparse_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_size = _get_hidden_size(self, hidden_states)
    gate_output = self.gate(hidden_states.reshape(-1, hidden_size))

    if isinstance(gate_output, tuple):
        # Transformers 5.x: gate is a router module and returns
        # (router_logits, routing_weights, selected_experts).
        _, routing_weights, selected_experts = gate_output
        return _qwen3_vl_moe_forward_transformers_5(self, hidden_states, routing_weights, selected_experts)

    # Transformers 4.57.x: gate is nn.Linear and experts use the old
    # (hidden_states, routing_weights, router_indices) call order.
    return _qwen3_vl_moe_forward_transformers_457(self, hidden_states, gate_output)


QWEN3_VL_MOE_PATCHES = {
    'Qwen3VLMoeTextExperts.forward': npu_packed_moe_experts_forward,
    'Qwen3VLMoeTextSparseMoeBlock.forward': npu_qwen3_vl_moe_sparse_block_forward,
    'Qwen3VLMoeTextRMSNorm': NpuRMSNorm,
    'apply_rotary_pos_emb': npu_apply_rotary_pos_emb,
}

# ---------------------------------------------------------------------------
# Qwen3.5-MoE patch
# ---------------------------------------------------------------------------


def _add_shared_expert(self, hidden_states: torch.Tensor, expert_output: torch.Tensor) -> torch.Tensor:
    if not (hasattr(self, 'shared_expert') and hasattr(self, 'shared_expert_gate')):
        return expert_output

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
    return expert_output + shared_expert_output


def _qwen3_5_moe_forward_transformers_5(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor,
                                        selected_experts: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    expert_output = self.experts(hidden_states, selected_experts, routing_weights)
    expert_output = _add_shared_expert(self, hidden_states, expert_output)
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)


def _qwen3_5_moe_forward_linear_gate(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    routing_weights, selected_experts = _topk_from_router_logits(self, hidden_states, router_logits)
    expert_output = self.experts(hidden_states, selected_experts, routing_weights)
    expert_output = _add_shared_expert(self, hidden_states, expert_output)
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)


def npu_qwen3_5_moe_sparse_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_dim = hidden_states.shape[-1]
    gate_output = self.gate(hidden_states.view(-1, hidden_dim))

    if isinstance(gate_output, tuple):
        # Transformers 5.x: Qwen3.5-MoE has packed experts plus shared expert.
        _, routing_weights, selected_experts = gate_output
        return _qwen3_5_moe_forward_transformers_5(self, hidden_states, routing_weights, selected_experts)

    return _qwen3_5_moe_forward_linear_gate(self, hidden_states, gate_output)


QWEN3_5_MOE_PATCHES = {
    'Qwen3_5MoeRMSNorm': NpuQwen3_5RMSNorm,
    'apply_rotary_pos_emb': npu_apply_rotary_pos_emb_qwen3_5,
    'Qwen3_5MoeMLP.forward': npu_swiglu_forward,
    'Qwen3_5MoeExperts.forward': npu_packed_moe_experts_forward,
    'Qwen3_5MoeSparseMoeBlock.forward': npu_qwen3_5_moe_sparse_block_forward,
}

QWEN3_5_MOE_OPTIONAL_PATCHES = {}

# ---------------------------------------------------------------------------
# Patch table and apply entry
# ---------------------------------------------------------------------------


def _build_patch_map(root, patches: dict[str, object], optional_patches: dict[str, object] | None = None):
    patch_map = dict(patches)
    for path, value in (optional_patches or {}).items():
        current = root
        for part in path.split('.'):
            if not hasattr(current, part):
                break
            current = getattr(current, part)
        else:
            patch_map[path] = value
    return patch_map


_APPLIED = False


def apply_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return

    patch_groups = [
        ('qwen2', modeling_qwen2, QWEN2_PATCHES, {}),
        ('qwen3', modeling_qwen3, QWEN3_PATCHES, {}),
        ('qwen3_moe', modeling_qwen3_moe, QWEN3_MOE_PATCHES, QWEN3_MOE_TRANSFORMERS_5_PATCHES),
        ('qwen3_vl_moe', modeling_qwen3_vl_moe, QWEN3_VL_MOE_PATCHES, {}),
    ]

    modeling_qwen3_5 = import_optional_module('transformers.models.qwen3_5.modeling_qwen3_5')
    modeling_qwen3_5_moe = import_optional_module('transformers.models.qwen3_5_moe.modeling_qwen3_5_moe')
    if modeling_qwen3_5 is not None:
        _patch_transformers_flash_linear_attention_available()
        patch_qwen3_5_chunk_gated_delta_rule_with_mindspeed()

    if modeling_qwen3_5 is not None:
        patch_groups.append(('qwen3_5', modeling_qwen3_5, QWEN3_5_PATCHES, {}))

    if modeling_qwen3_5_moe is not None:
        patch_groups.append(('qwen3_5_moe', modeling_qwen3_5_moe, QWEN3_5_MOE_PATCHES, QWEN3_5_MOE_OPTIONAL_PATCHES))

    for _group_name, module, patches, optional_patches in patch_groups:
        apply_patch_map(module, _build_patch_map(module, patches, optional_patches))

    _APPLIED = True

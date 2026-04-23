# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import accelerate.utils.fsdp_utils as fsdp_utils
import fcntl
import importlib
import os
import torch
import torch.nn.functional as F
import torch_npu
from accelerate.accelerator import Accelerator
from functools import wraps
from torch import nn
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()

_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT = '600'
_DEFAULT_TRITON_ASCEND_LAUNCHER_LOCK_PATH = '/tmp/swift_triton_ascend_launcher_compile.lock'
_ORIGINAL_MINDSPEED_TE_CP_CLASS = None


def _set_default_hccl_connect_timeout_for_npu() -> None:
    if 'HCCL_CONNECT_TIMEOUT' in os.environ:
        return

    os.environ['HCCL_CONNECT_TIMEOUT'] = _DEFAULT_NPU_HCCL_CONNECT_TIMEOUT
    logger.info(f'Set HCCL_CONNECT_TIMEOUT={_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT} by default for NPU.')


_set_default_hccl_connect_timeout_for_npu()


def _import_optional_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        logger.debug('Failed to import optional module %s: %s', module_name, exc)
        return None

def _patch_triton_ascend_launcher_compile_lock() -> None:
    ascend_driver = _import_optional_module('triton.backends.ascend.driver')
    if ascend_driver is None:
        return

    make_launcher_stub = getattr(ascend_driver, 'make_npu_launcher_stub', None)
    if make_launcher_stub is None or getattr(make_launcher_stub, '_swift_compile_lock', False):
        return

    @wraps(make_launcher_stub)
    def _locked_make_npu_launcher_stub(*args, **kwargs):
        lock_path = os.environ.get('SWIFT_TRITON_ASCEND_LAUNCHER_LOCK',
                                   _DEFAULT_TRITON_ASCEND_LAUNCHER_LOCK_PATH)
        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                return make_launcher_stub(*args, **kwargs)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    _locked_make_npu_launcher_stub._swift_compile_lock = True
    ascend_driver.make_npu_launcher_stub = _locked_make_npu_launcher_stub
    logger.info(
        'Patched Ascend Triton launcher compilation with file lock: %s.',
        os.environ.get('SWIFT_TRITON_ASCEND_LAUNCHER_LOCK', _DEFAULT_TRITON_ASCEND_LAUNCHER_LOCK_PATH),
    )


def _patch_transformers_flash_linear_attention_available() -> None:
    def _is_flash_linear_attention_available() -> bool:
        return True

    transformers_utils = _import_optional_module('transformers.utils')
    if transformers_utils is not None:
        setattr(transformers_utils, 'is_flash_linear_attention_available', _is_flash_linear_attention_available)

    transformers_import_utils = _import_optional_module('transformers.utils.import_utils')
    if transformers_import_utils is not None:
        setattr(transformers_import_utils, 'is_flash_linear_attention_available',
                _is_flash_linear_attention_available)


def _should_fallback_to_torch_chunk_gated_delta_rule(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False

    message = str(exc)
    fallback_patterns = (
        'Failed to compile',
        'launcher_cxx11abi1.cxx',
        'internal compiler error',
        'wide_int_to_tree_1',
        'ConvertLinalgRToBinary',
        'hivm.hir',
    )
    return any(pattern in message for pattern in fallback_patterns)


def _build_qwen3_5_chunk_gated_delta_rule_with_fallback(module_name: str, module: Any,
                                                        mindspeed_chunk_gated_delta_rule):
    torch_chunk_gated_delta_rule = getattr(module, 'torch_chunk_gated_delta_rule', None)
    if torch_chunk_gated_delta_rule is None:
        return mindspeed_chunk_gated_delta_rule

    warned_runtime_fallback = False
    warned_forced_fallback = False

    @wraps(mindspeed_chunk_gated_delta_rule)
    def _wrapped_chunk_gated_delta_rule(*args, **kwargs):
        nonlocal warned_runtime_fallback, warned_forced_fallback

        if os.environ.get('SWIFT_QWEN3_5_CHUNK_GATED_DELTA_RULE_FORCE_TORCH', '0') == '1':
            if not warned_forced_fallback:
                logger.warning('Using torch_chunk_gated_delta_rule for %s because '
                               'SWIFT_QWEN3_5_CHUNK_GATED_DELTA_RULE_FORCE_TORCH=1.', module_name)
                warned_forced_fallback = True
            return torch_chunk_gated_delta_rule(*args, **kwargs)

        if os.environ.get('SWIFT_QWEN3_5_CHUNK_GATED_DELTA_RULE_FORCE_TRITON', '0') == '1':
            return mindspeed_chunk_gated_delta_rule(*args, **kwargs)

        try:
            return mindspeed_chunk_gated_delta_rule(*args, **kwargs)
        except Exception as exc:
            if not _should_fallback_to_torch_chunk_gated_delta_rule(exc):
                raise
            if not warned_runtime_fallback:
                logger.warning('Falling back to torch_chunk_gated_delta_rule for %s after MindSpeed '
                               'chunk_gated_delta_rule failed: %s', module_name, exc)
                warned_runtime_fallback = True
            return torch_chunk_gated_delta_rule(*args, **kwargs)

    _wrapped_chunk_gated_delta_rule._swift_chunk_gated_delta_rule_fallback = True
    return _wrapped_chunk_gated_delta_rule


def patch_qwen3_5_chunk_gated_delta_rule_with_mindspeed() -> None:
    try:
        from .chunk_gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        logger.warning('Failed to import embedded MindSpeed chunk_gated_delta_rule: %s', exc)
        return

    patched_modules = []
    for module_name in ('transformers.models.qwen3_5.modeling_qwen3_5',
                        'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
                        'transformers.models.qwen3_next.modeling_qwen3_next'):
        module = _import_optional_module(module_name)
        if module is None:
            continue

        setattr(module, 'is_flash_linear_attention_available', lambda: True)
        setattr(module, 'is_fast_path_available', True)
        # FLA's fused RMSNormGated initializes with torch.cuda.current_device(),
        # so keep the native Qwen3.5 torch implementation on NPU.
        setattr(module, 'FusedRMSNormGated', None)
        setattr(
            module, 'chunk_gated_delta_rule',
            _build_qwen3_5_chunk_gated_delta_rule_with_fallback(module_name, module, chunk_gated_delta_rule))
        patched_modules.append(module_name)

    if patched_modules:
        logger.info('Patched Qwen3.5 chunk_gated_delta_rule to embedded MindSpeed implementation: %s.',
                    ', '.join(patched_modules))


_patch_triton_ascend_launcher_compile_lock()
_patch_transformers_flash_linear_attention_available()
patch_qwen3_5_chunk_gated_delta_rule_with_mindspeed()
modeling_qwen3_5 = _import_optional_module('transformers.models.qwen3_5.modeling_qwen3_5')


def patch_mindspeed_te_cp_implementation(megatron_args: dict[str, Any]) -> None:
    """
    Route NPU CP to the legacy MindSpeed TE adaptor when the new strategy factory
    only supports kvallgather.
    """
    try:
        import mindspeed.te.pytorch.attention.dot_product_attention.dot_product_attention as ms_te_dpa
        from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
    except ImportError as e:
        logger.warning(f'Failed to import MindSpeed CP modules before repatch: {e}')
        return

    global _ORIGINAL_MINDSPEED_TE_CP_CLASS
    if _ORIGINAL_MINDSPEED_TE_CP_CLASS is None:
        _ORIGINAL_MINDSPEED_TE_CP_CLASS = getattr(ms_te_dpa, 'MindSpeedTEDotProductAttention', None)

    if _ORIGINAL_MINDSPEED_TE_CP_CLASS is None:
        logger.warning('MindSpeedTEDotProductAttention is unavailable before repatch; skip CP workaround.')
        return

    cp_algo = megatron_args.get('context_parallel_algo', 'megatron_cp_algo')
    use_legacy_cp_te = int(megatron_args.get('context_parallel_size', 1)) > 1 and cp_algo != 'kvallgather_cp_algo'
    target_cls = MindSpeedCPDotProductAttention if use_legacy_cp_te else _ORIGINAL_MINDSPEED_TE_CP_CLASS

    if getattr(ms_te_dpa, 'MindSpeedTEDotProductAttention', None) is target_cls:
        return

    ms_te_dpa.MindSpeedTEDotProductAttention = target_cls
    logger.info(
        'Patched MindSpeedTEDotProductAttention to %s for context_parallel_size=%s, context_parallel_algo=%s.',
        target_cls.__name__,
        megatron_args.get('context_parallel_size', 1),
        cp_algo,
    )


class NPUCastError(RuntimeError):
    """Raised when fp32 casting fails during NPU FSDP2 preparation."""


def _get_first_parameter(module: torch.nn.Module) -> torch.nn.Parameter | None:
    for param in module.parameters(recurse=True):
        return param
    return None


def _needs_fp32_cast_for_npu(
    module: torch.nn.Module,
    accelerator: Accelerator,
) -> bool:
    if accelerator.device.type != 'npu':
        return False

    param = _get_first_parameter(module)
    if param is None:
        return False

    return param.is_floating_point() and param.dtype != torch.float32


def _cast_to_fp32(module: torch.nn.Module) -> torch.nn.Module:
    """
    Cast module parameters to fp32.

    Assumes parameters are already on CPU or meta device.
    Only dtype is changed; device is preserved.
    """
    try:
        return module.to(torch.float32)
    except Exception as exc:
        raise NPUCastError(f'Failed to cast {module.__class__.__name__} to fp32.') from exc


# ----------------------------------------------------------------------
# Patch accelerate.utils.fsdp_utils.fsdp2_prepare_model
# ----------------------------------------------------------------------

_original_fsdp2_prepare_model = fsdp_utils.fsdp2_prepare_model


@wraps(_original_fsdp2_prepare_model)
def wrapped_fsdp2_prepare_model(
    accelerator: Accelerator,
    model: torch.nn.Module,
):
    if _needs_fp32_cast_for_npu(model, accelerator):
        model = _cast_to_fp32(model)

    return _original_fsdp2_prepare_model(accelerator, model)


fsdp_utils.fsdp2_prepare_model = wrapped_fsdp2_prepare_model

# ----------------------------------------------------------------------
# Patch Accelerator._prepare_fsdp2
# ----------------------------------------------------------------------

_original_prepare_fsdp2 = Accelerator._prepare_fsdp2


@wraps(_original_prepare_fsdp2)
def wrapped_prepare_fsdp2(
    self: Accelerator,
    *args,
    **kwargs,
):
    patched_args = [
        _cast_to_fp32(obj) if isinstance(obj, torch.nn.Module) and _needs_fp32_cast_for_npu(obj, self) else obj
        for obj in args
    ]

    return _original_prepare_fsdp2(self, *patched_args, **kwargs)


Accelerator._prepare_fsdp2 = wrapped_prepare_fsdp2


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


def npu_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    # Concat all weights
    input_dtype = hidden_states.dtype
    up_weight_list = [e.up_proj.weight.t().to(input_dtype) for e in self.experts]
    gate_weight_list = [e.gate_proj.weight.t().to(input_dtype) for e in self.experts]
    down_weight_list = [e.down_proj.weight.t().to(input_dtype) for e in self.experts]
    w1 = torch.stack(up_weight_list)
    w2 = torch.stack(gate_weight_list)
    w3 = torch.stack(down_weight_list)

    # Copied from mindspeed moe_utils.py:permute
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

    probs = routing_weights
    num_unpermuted_tokens = probs.numel()
    topk = self.top_k
    permuted_tokens = down_res

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    final_hidden_states = unpermuted_tokens.sum(dim=1).to(hidden_states.dtype)
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    return final_hidden_states, router_logits


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


class NpuMoeFused:

    @staticmethod
    def npu_moe_experts_forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor,
                                router_indices: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states,
                                                                              router_indices.to(torch.int32))
        tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate_hidden_states = GmmFunction.apply(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = GmmFunction.apply(intermediate_activations, self.down_proj, tokens_per_expert)
        next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states

    @staticmethod
    def npu_moe_sparse_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, routing_weights, router_indices)
        return routed_out


def _setattr_path(root: Any, path: str, value: Any) -> None:
    current = root
    parts = path.split('.')
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def _apply_patch_map(root: Any, patch_map: dict[str, Any]) -> None:
    for path, value in patch_map.items():
        _setattr_path(root, path, value)


_QWEN3_5_PATCH_TABLE: tuple[tuple[Any, dict[str, Any]], ...] = ()
if modeling_qwen3_5 is not None:
    _QWEN3_5_PATCH_TABLE = ((modeling_qwen3_5, {
        'Qwen3_5RMSNorm': NpuQwen3_5RMSNorm,
        'apply_rotary_pos_emb': npu_apply_rotary_pos_emb_qwen3_5,
        'Qwen3_5MLP.forward': npu_swiglu_forward,
    }), )

_PATCH_TABLE: tuple[tuple[Any, dict[str, Any]], ...] = (
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
    *_QWEN3_5_PATCH_TABLE,
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

for _module, _patch_map in _PATCH_TABLE:
    _apply_patch_map(_module, _patch_map)

# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import copy
import importlib
import inspect
import sys
import torch
from functools import wraps
from types import ModuleType
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()

_ORIGINAL_MINDSPEED_TE_CP_CLASS = None
_ORIGINAL_MINDSPEED_GDN = None
_FLA_GDN_PATCH_TARGET = 'fla.ops.gated_delta_rule.chunk_gated_delta_rule'
_MINDSPEED_FSDP_GRAD_REDUCE_MODULE = 'mindspeed.core.distributed.custom_fsdp.param_and_grad_buffer'


def _wrap_mindspeed_fsdp_gradient_reduce(original_gradient_reduce):
    """Treat a missing Megatron-FSDP gradient scale as the multiplicative identity."""

    @wraps(original_gradient_reduce)
    def gradient_reduce_with_none_scaling(self, bucket_group, *args, **kwargs):
        restored_buffers = []
        seen_buffers = set()
        for bucket_id in bucket_group:
            grad_buffer = self.get_fsdp_buffer(bucket_id)
            buffer_id = id(grad_buffer)
            if buffer_id in seen_buffers or grad_buffer.gradient_scaling_factor is not None:
                continue
            seen_buffers.add(buffer_id)
            original_ddp_config = grad_buffer.ddp_config
            restored_buffers.append((grad_buffer, original_ddp_config))
            grad_buffer.gradient_scaling_factor = 1.0

            # `None` means SUM without any pre-scaling. Keep that semantic even if
            # average-in-collective is enabled, without changing the shared config
            # subsequently used by the outer-HSDP reduction.
            if original_ddp_config.average_in_collective:
                grad_buffer.ddp_config = copy.copy(original_ddp_config)
                grad_buffer.ddp_config.average_in_collective = False

        try:
            return original_gradient_reduce(self, bucket_group, *args, **kwargs)
        finally:
            for grad_buffer, original_ddp_config in restored_buffers:
                grad_buffer.gradient_scaling_factor = None
                grad_buffer.ddp_config = original_ddp_config

    gradient_reduce_with_none_scaling._swift_handles_none_gradient_scaling = True
    return gradient_reduce_with_none_scaling


def patch_mindspeed_megatron_fsdp_gradient_scaling(megatron_args: dict[str, Any]) -> None:
    """Patch the MindSpeed 0.16 Megatron-FSDP reducer that multiplies by ``None``."""
    if not megatron_args.get('use_megatron_fsdp', False):
        return

    grad_buffer_module = importlib.import_module(
        'megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer')
    grad_reduce = grad_buffer_module.GradReducePipeline._bucket_group_gradient_reduce
    if getattr(grad_reduce, '_swift_handles_none_gradient_scaling', False):
        return
    if grad_reduce.__module__ != _MINDSPEED_FSDP_GRAD_REDUCE_MODULE:
        logger.info('MindSpeed Megatron-FSDP gradient reducer is not active; skip the None-scaling patch.')
        return

    try:
        source = inspect.getsource(grad_reduce)
    except (OSError, TypeError) as e:
        logger.warning('Cannot inspect the active MindSpeed Megatron-FSDP gradient reducer; skip patch: %s', e)
        return
    has_unconditional_scale = 'bucket.data.mul_(scaling_factor)' in source
    has_none_guard = 'if scaling_factor is None' in source or 'if scaling_factor is not None' in source
    if not has_unconditional_scale or has_none_guard:
        logger.info('MindSpeed Megatron-FSDP gradient reducer already handles None scaling; skip patch.')
        return

    grad_buffer_module.GradReducePipeline._bucket_group_gradient_reduce = _wrap_mindspeed_fsdp_gradient_reduce(
        grad_reduce)
    logger.info('Patched MindSpeed Megatron-FSDP gradient reduction to handle scaling_factor=None.')


def _get_mindspeed_fsdp_model_parameter(model, param_name):
    model_param_name = param_name[len('module.'):] if param_name.startswith('module.') else param_name
    num_experts = getattr(getattr(model, 'config', None), 'num_moe_experts', None)
    if num_experts:
        from megatron.core.transformer.fsdp_dtensor_checkpoint import expert_param_local_key
        model_param_name = expert_param_local_key(model_param_name, num_experts)
    return model.get_parameter(model_param_name)


def _build_empty_optimizer_state(state_template, template_param, dist_param, param_name):
    missing_state = {}
    for state_key, template_value in state_template.items():
        if not isinstance(template_value, torch.Tensor):
            missing_state[state_key] = copy.deepcopy(template_value)
            continue

        if template_value.ndim == 0 or state_key == 'step':
            missing_state[state_key] = template_value.detach().clone()
            continue

        if tuple(template_value.shape) != tuple(template_param.shape):
            raise RuntimeError(
                f'Cannot infer the empty Megatron-FSDP optimizer state `{state_key}` for `{param_name}`: '
                f'template state shape {tuple(template_value.shape)} does not match its parameter shape '
                f'{tuple(template_param.shape)}.')
        missing_state[state_key] = torch.zeros_like(dist_param, dtype=template_value.dtype)
    return missing_state


def complete_mindspeed_fsdp_dtensor_optimizer_state(state_dict, model) -> None:
    """Add optimizer shards omitted by MindSpeed so all DP ranks use the same state keys."""
    optimizer_state_dict = state_dict.get('optimizer')
    if not optimizer_state_dict:
        return
    optimizer_state = optimizer_state_dict.get('state', {})
    param_to_group_meta = optimizer_state_dict.get('param_to_group_meta', {})
    if not param_to_group_meta:
        return
    if set(param_to_group_meta).issubset(optimizer_state):
        optimizer_state_dict['state'] = {
            **{
                key: optimizer_state[key]
                for key in param_to_group_meta
            },
            **{
                key: value
                for key, value in optimizer_state.items() if key not in param_to_group_meta
            },
        }
        return
    if not optimizer_state:
        raise RuntimeError('Cannot infer Megatron-FSDP optimizer state fields from an empty local state dict.')

    template_param_name = next(
        (name for name in param_to_group_meta if name in optimizer_state and optimizer_state[name]), None)
    if template_param_name is None:
        raise RuntimeError('Cannot match a Megatron-FSDP optimizer state template to a model parameter.')
    state_template = optimizer_state[template_param_name]
    template_param = _get_mindspeed_fsdp_model_parameter(model, template_param_name)
    completed_state = {}
    added_count = 0
    for param_name in param_to_group_meta:
        if param_name in optimizer_state:
            completed_state[param_name] = optimizer_state[param_name]
            continue

        dist_param = _get_mindspeed_fsdp_model_parameter(model, param_name)
        local_param = dist_param.to_local() if hasattr(dist_param, 'to_local') else dist_param
        if local_param.numel() != 0:
            raise RuntimeError(
                f'MindSpeed omitted Megatron-FSDP optimizer state for `{param_name}`, but its local parameter shard '
                f'is not empty (numel={local_param.numel()}). Refusing to replace a non-empty optimizer shard with '
                f'zeros.')
        completed_state[param_name] = _build_empty_optimizer_state(
            state_template,
            template_param,
            dist_param,
            param_name,
        )
        added_count += 1

    completed_state.update({key: value for key, value in optimizer_state.items() if key not in completed_state})
    optimizer_state_dict['state'] = completed_state
    logger.info('Added %d empty local optimizer shards for MindSpeed Megatron-FSDP checkpointing.', added_count)


def load_mindspeed_fsdp_dtensor_optimizer_state_dict(distributed_optimizers, state_dict) -> bool:
    """Restore an FSDP DTensor optimizer state without MindSpeed's legacy checkpoint loader."""
    is_fsdp_dtensor_state = isinstance(state_dict,
                                       dict) and 'state' in state_dict and 'param_to_group_meta' in state_dict
    if not is_fsdp_dtensor_state:
        return False
    if len(distributed_optimizers) != 1:
        raise RuntimeError(f'MindSpeed FSDP DTensor optimizer compatibility supports exactly one distributed '
                           f'optimizer, got {len(distributed_optimizers)}.')

    distributed_optimizer = distributed_optimizers[0]
    logger.warning('Loading FSDP DTensor optimizer state with the Megatron-Core compatibility path because '
                   'MindSpeed DistributedOptimizer.load_state_dict expects the legacy checkpoint structure.')
    inner_state_dict = dict(state_dict)
    inner_state_dict['param_groups'] = distributed_optimizer._param2group_meta_to_param_groups(
        inner_state_dict.pop('param_to_group_meta'), distributed_optimizer.optimizer.param_groups)
    distributed_optimizer.optimizer.load_state_dict(inner_state_dict)
    return True


def _mindspeed_gdn_with_safe_varlen(q,
                                    k,
                                    v,
                                    g,
                                    beta,
                                    scale=None,
                                    initial_state=None,
                                    output_final_state=False,
                                    use_qk_l2norm_in_kernel=False,
                                    cu_seqlens=None,
                                    chunk_size=64,
                                    head_first=False):
    kwargs = {
        'scale': scale,
        'output_final_state': output_final_state,
        'use_qk_l2norm_in_kernel': use_qk_l2norm_in_kernel,
        'chunk_size': chunk_size,
        'head_first': head_first,
    }
    if cu_seqlens is None:
        return _ORIGINAL_MINDSPEED_GDN(q, k, v, g, beta, initial_state=initial_state, **kwargs)

    # MindSpeed's arch35 varlen backward uses the local sequence length as the packed gate stride.
    # Keep the same implementation but run each sequence independently to avoid the invalid indexing.
    import torch
    sequence_dim = 2 if head_first else 1
    offsets = cu_seqlens.detach().cpu().tolist()
    outputs, final_states = [], []
    for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
        length = end - start
        inputs = [x.narrow(sequence_dim, start, length) for x in (q, k, v, g, beta)]
        state = None if initial_state is None else initial_state[i:i + 1]
        output, final_state = _ORIGINAL_MINDSPEED_GDN(*inputs, initial_state=state, **kwargs)
        outputs.append(output)
        if output_final_state:
            final_states.append(final_state)
    output = torch.cat(outputs, dim=sequence_dim)
    final_state = torch.cat(final_states) if output_final_state else None
    return output, final_state


def prepare_mindspeed_gdn_import() -> None:
    try:
        import fla.utils
    except ModuleNotFoundError as e:
        if e.name not in {'fla', 'fla.utils'}:
            raise
        gdn_module = ModuleType('mindspeed.core.ssm.chunk_gated_delta_rule')

        def torch_chunk_gated_delta_rule(q,
                                         k,
                                         v,
                                         g,
                                         beta,
                                         scale=None,
                                         initial_state=None,
                                         output_final_state=False,
                                         use_qk_l2norm_in_kernel=False,
                                         cu_seqlens=None,
                                         chunk_size=64,
                                         head_first=False,
                                         **kwargs):
            if cu_seqlens is not None:
                raise ValueError('Torch GDN fallback does not support cu_seqlens.')
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule as torch_gdn
            return torch_gdn(
                q,
                k,
                v,
                g=g,
                beta=beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        gdn_module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        gdn_module._ms_swift_torch_fallback = True
        sys.modules[gdn_module.__name__] = gdn_module
    else:
        import torch_npu
        device_name = torch_npu.npu.get_device_name()
        # MindSpeed still imports this flag after it was removed from upstream FLA.
        if not hasattr(fla.utils, 'USE_CUDA_GRAPH'):
            if 'Ascend910_95' in device_name or 'Ascend950' in device_name:
                fla.utils.USE_CUDA_GRAPH = False


def _apply_gdn_patch(MindSpeedPatchesManager, patch, implementation) -> None:
    if patch is not None:
        MindSpeedPatchesManager.register_patch(
            _FLA_GDN_PATCH_TARGET,
            implementation,
            force_patch=True,
        )
        MindSpeedPatchesManager.apply_patches()
    else:
        try:
            fla_gated_delta_rule = importlib.import_module('fla.ops.gated_delta_rule')
        except Exception:
            pass
        else:
            fla_gated_delta_rule.chunk_gated_delta_rule = implementation

    # mcore-bridge may have cached the callable before a runtime repatch.
    bridge_gdn = sys.modules.get('mcore_bridge.model.modules.gated_delta_net')
    if bridge_gdn is not None:
        bridge_gdn.chunk_gated_delta_rule = implementation

    if patch is not None:
        fla_gated_delta_rule = importlib.import_module('fla.ops.gated_delta_rule')
        if fla_gated_delta_rule.chunk_gated_delta_rule is not implementation:
            raise RuntimeError('MindSpeed did not install the selected Megatron GDN implementation.')
    if bridge_gdn is not None and bridge_gdn.chunk_gated_delta_rule is not implementation:
        raise RuntimeError('Failed to refresh the mcore-bridge cached GDN implementation.')


def _patch_mindspeed_fla_gdn_implementation(MindSpeedPatchesManager) -> None:
    patch = MindSpeedPatchesManager.patches_info.get(_FLA_GDN_PATCH_TARGET)

    mindspeed_gdn_module = sys.modules.get('mindspeed.core.ssm.chunk_gated_delta_rule')
    if getattr(mindspeed_gdn_module, '_ms_swift_torch_fallback', False):
        torch_gdn = mindspeed_gdn_module.chunk_gated_delta_rule
        _apply_gdn_patch(MindSpeedPatchesManager, patch, torch_gdn)
        logger.info('Using torch chunk_gated_delta_rule for Megatron GDN because FLA is unavailable.')
        return

    import torch_npu
    device_name = torch_npu.npu.get_device_name()
    if 'Ascend910_95' in device_name or 'Ascend950' in device_name:
        from mindspeed.core.ssm.chunk_gated_delta_rule import chunk_gated_delta_rule as mindspeed_gdn
        global _ORIGINAL_MINDSPEED_GDN
        if _ORIGINAL_MINDSPEED_GDN is None:
            _ORIGINAL_MINDSPEED_GDN = mindspeed_gdn
        _apply_gdn_patch(MindSpeedPatchesManager, patch, _mindspeed_gdn_with_safe_varlen)
        logger.info(
            'Using MindSpeed chunk_gated_delta_rule with safe varlen fallback for Megatron GDN on Ascend arch35.')
        return

    fla_error = None
    if (patch is not None and patch.orig_func is not None and patch.orig_func.__module__.startswith('fla.')):
        # MindSpeed propagates its replacement into already imported submodules,
        # so importing from ``fla.ops.gated_delta_rule.chunk`` again is not enough.
        fla_chunk_gated_delta_rule = patch.orig_func
    else:
        try:
            from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
        except Exception as e:
            fla_chunk_gated_delta_rule = None
            fla_error = e

    if fla_chunk_gated_delta_rule is not None:
        try:
            if not fla_chunk_gated_delta_rule.__module__.startswith('fla.'):
                raise RuntimeError('resolved a non-FLA callable: '
                                   f'{fla_chunk_gated_delta_rule.__module__}.'
                                   f'{fla_chunk_gated_delta_rule.__name__}')
            _apply_gdn_patch(MindSpeedPatchesManager, patch, fla_chunk_gated_delta_rule)
            logger.info(
                'Using upstream FLA chunk_gated_delta_rule for Megatron GDN: module=%s, source=%s.',
                fla_chunk_gated_delta_rule.__module__,
                inspect.getsourcefile(inspect.unwrap(fla_chunk_gated_delta_rule)),
            )
            return
        except Exception as e:
            fla_error = e

    logger.warning(
        'FLA GDN is unavailable (%s); keep the current MindSpeed/Megatron GDN implementation unchanged. '
        'If it does not support packed cu_seqlens, the GDN call will fail at runtime.',
        fla_error,
    )


def patch_mindspeed_fla_gdn_implementation() -> None:
    """Use torch GDN without FLA, MindSpeed GDN on arch35, and upstream FLA elsewhere."""
    from mindspeed.patch_utils import MindSpeedPatchesManager

    try:
        _patch_mindspeed_fla_gdn_implementation(MindSpeedPatchesManager)
    except Exception as e:
        logger.warning('Failed to apply the optional FLA GDN patch; keep the current implementation: %s', e)


def patch_mindspeed_te_cp_implementation(megatron_args: dict[str, Any]) -> None:
    """
    Route NPU CP to the legacy MindSpeed TE adaptor when the new strategy factory
    only supports kvallgather.
    """
    # MindSpeed 0.15.3 replaced the TE context-parallel attention class with a
    # new implementation. That new class does not yet cover all CP algorithms,
    # so the default non-kvallgather path can fail during Megatron training.
    # For those algorithms, temporarily route TE attention back to the legacy
    # MindSpeedCPDotProductAttention adaptor. Once MindSpeed's new CP class has
    # feature parity, this compatibility patch can be removed.
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


def patch_mindspeed_te_layernorm_linear_frozen_weight() -> None:
    """Route frozen MindSpeed TE LayerNormLinear weights through Megatron's frozen-weight path."""
    try:
        ms_te_layernorm_linear = importlib.import_module('mindspeed.te.pytorch.module.layernorm_column_parallel_linear')
        from megatron.core.tensor_parallel.layers import linear_with_frozen_weight
    except ImportError as e:
        logger.warning('Failed to import MindSpeed TE LayerNormLinear modules: %s', e)
        return

    linear_impl_name = 'linear_with_grad_accumulation_and_async_allreduce'
    trainable_weight_impl = getattr(ms_te_layernorm_linear, linear_impl_name, None)
    if trainable_weight_impl is None:
        logger.warning('MindSpeed TE LayerNormLinear does not expose %s; skip frozen-weight patch.', linear_impl_name)
        return
    if getattr(trainable_weight_impl, '_swift_supports_frozen_weight', False):
        return

    @wraps(trainable_weight_impl)
    def linear_with_frozen_weight_dispatch(
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer=None,
        wgrad_deferral_limit=0,
        async_grad_allreduce=None,
        tp_group=None,
    ):
        if weight.requires_grad:
            return trainable_weight_impl(
                input=input,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                allreduce_dgrad=allreduce_dgrad,
                sequence_parallel=sequence_parallel,
                grad_output_buffer=grad_output_buffer,
                wgrad_deferral_limit=wgrad_deferral_limit,
                async_grad_allreduce=async_grad_allreduce,
                tp_group=tp_group,
            )
        return linear_with_frozen_weight(
            input=input,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=sequence_parallel,
            async_grad_allreduce=async_grad_allreduce,
            tp_group=tp_group,
        )

    linear_with_frozen_weight_dispatch._swift_supports_frozen_weight = True
    setattr(ms_te_layernorm_linear, linear_impl_name, linear_with_frozen_weight_dispatch)
    logger.info('Patched MindSpeed TE LayerNormLinear to use Megatron frozen-weight backward for frozen weights.')


def patch_mindspeed_te_grouped_linear_save_original_input() -> None:
    """Allow BF16 MindSpeed grouped linear layers to use Megatron activation offloading."""
    te_extension = importlib.import_module('megatron.core.extensions.transformer_engine')
    set_save_original_input = te_extension.set_save_original_input
    if getattr(set_save_original_input, '_swift_supports_mindspeed_bf16_grouped_linear', False):
        return

    @wraps(set_save_original_input)
    def set_save_original_input_dispatch(module):
        config = getattr(module, 'config', None)
        is_mindspeed_grouped_linear = type(module).__module__ == 'mindspeed.te.pytorch.module.grouped_linear'
        uses_quantized_tensors = bool(getattr(config, 'fp8', None) or getattr(config, 'fp4', None))
        if is_mindspeed_grouped_linear and not uses_quantized_tensors:
            # MindSpeed's BF16 TEGroupedLinearGMM saves the original input tensor directly.
            return
        return set_save_original_input(module)

    set_save_original_input_dispatch._swift_supports_mindspeed_bf16_grouped_linear = True
    te_extension.set_save_original_input = set_save_original_input_dispatch
    logger.info('Patched Megatron set_save_original_input for MindSpeed BF16 grouped linear layers.')


def patch_mindspeed_gdn_cp_helpers(megatron_args: dict[str, Any]) -> None:
    """Expose MindSpeed's backported GDN CP helpers to Megatron Core versions before 0.18."""
    if int(megatron_args.get('context_parallel_size', 1)) <= 1:
        return

    megatron_gdn = importlib.import_module('megatron.core.ssm.gated_delta_net')
    mindspeed_gdn = importlib.import_module('mindspeed.core.ssm.gated_delta_net')
    patched_helpers = []
    for helper_name in ('tensor_a2a_cp2hp', 'tensor_a2a_hp2cp'):
        if hasattr(megatron_gdn, helper_name):
            continue
        helper = getattr(mindspeed_gdn, helper_name, None)
        if helper is None:
            raise RuntimeError(f'MindSpeed does not provide the required GDN CP helper: {helper_name}.')
        setattr(megatron_gdn, helper_name, helper)
        patched_helpers.append(helper_name)

    if patched_helpers:
        logger.info('Patched Megatron GDN CP helpers from MindSpeed: %s.', ', '.join(patched_helpers))


def apply_mindspeed_patches(megatron_args: dict[str, Any]) -> None:
    """Apply MindSpeed compatibility patches around its runtime repatch in the required order."""
    from mindspeed.megatron_adaptor import repatch

    patch_mindspeed_te_cp_implementation(megatron_args)
    repatch(megatron_args)
    patch_mindspeed_megatron_fsdp_gradient_scaling(megatron_args)
    patch_mindspeed_gdn_cp_helpers(megatron_args)
    patch_mindspeed_te_layernorm_linear_frozen_weight()
    patch_mindspeed_te_grouped_linear_save_original_input()
    patch_mindspeed_fla_gdn_implementation()

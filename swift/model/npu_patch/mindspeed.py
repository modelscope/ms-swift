# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib
import inspect
import sys
from functools import wraps
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()

_ORIGINAL_MINDSPEED_TE_CP_CLASS = None
_FLA_GDN_PATCH_TARGET = 'fla.ops.gated_delta_rule.chunk_gated_delta_rule'


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
    """Best-effort preference for upstream FLA while preserving the current GDN implementation."""
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
        ms_te_layernorm_linear = importlib.import_module(
            'mindspeed.te.pytorch.module.layernorm_column_parallel_linear')
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


def apply_mindspeed_patches(megatron_args: dict[str, Any]) -> None:
    """Apply MindSpeed compatibility patches around its runtime repatch in the required order."""
    from mindspeed.megatron_adaptor import repatch

    patch_mindspeed_te_cp_implementation(megatron_args)
    repatch(megatron_args)
    patch_mindspeed_te_layernorm_linear_frozen_weight()
    patch_mindspeed_fla_gdn_implementation()

# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib
import inspect
import sys
from types import ModuleType
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()

_ORIGINAL_MINDSPEED_TE_CP_CLASS = None
_FLA_GDN_PATCH_TARGET = 'fla.ops.gated_delta_rule.chunk_gated_delta_rule'


def prepare_mindspeed_gdn_import() -> None:
    try:
        import fla.utils
    except ModuleNotFoundError as e:
        if e.name not in {'fla', 'fla.utils'}:
            raise
        gdn_module = ModuleType('mindspeed.core.ssm.chunk_gated_delta_rule')

        def torch_chunk_gated_delta_rule(
                q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, cu_seqlens=None, chunk_size=64, head_first=False, **kwargs):
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
        _apply_gdn_patch(MindSpeedPatchesManager, patch, mindspeed_gdn)
        logger.info(
            'Using MindSpeed chunk_gated_delta_rule for Megatron GDN on Ascend arch35: module=%s, source=%s.',
            mindspeed_gdn.__module__,
            inspect.getsourcefile(inspect.unwrap(mindspeed_gdn)),
        )
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

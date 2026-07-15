# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import functools
import importlib
import inspect
import sys
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()

_ORIGINAL_MINDSPEED_TE_CP_CLASS = None
_FLA_GDN_PATCH_TARGET = 'fla.ops.gated_delta_rule.chunk_gated_delta_rule'
_LOGGED_TORCH_GDN_FALLBACK = False


def _get_native_torch_gdn(patch, bridge_gdn):
    candidates = [
        getattr(bridge_gdn, 'torch_chunk_gated_delta_rule', None) if bridge_gdn is not None else None,
        getattr(patch, 'patch_func', None) if patch is not None else None,
    ]
    try:
        from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
        candidates.append(torch_chunk_gated_delta_rule)
    except ImportError:
        pass

    for candidate in candidates:
        if (candidate is not None and not candidate.__module__.startswith('fla.')
                and (candidate.__name__ == 'torch_chunk_gated_delta_rule'
                     or getattr(candidate, '_swift_native_torch_gdn', False))):
            return candidate


def _adapt_native_torch_gdn(native_torch_gdn):
    if getattr(native_torch_gdn, '_swift_accepts_cu_seqlens', False):
        return native_torch_gdn
    try:
        if 'cu_seqlens' in inspect.signature(native_torch_gdn).parameters:
            return native_torch_gdn
    except (TypeError, ValueError):
        pass

    @functools.wraps(native_torch_gdn)
    def native_torch_gdn_adapter(*args, cu_seqlens=None, **kwargs):
        if cu_seqlens is not None:
            raise RuntimeError('Packed Megatron GDN requires flash-linear-attention; '
                               'the native Torch fallback only supports non-packed sequences.')
        return native_torch_gdn(*args, **kwargs)

    native_torch_gdn_adapter._swift_accepts_cu_seqlens = True
    native_torch_gdn_adapter._swift_native_torch_gdn = True
    return native_torch_gdn_adapter


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
    bridge_gdn = sys.modules.get('mcore_bridge.model.modules.gated_delta_net')
    native_torch_gdn = _get_native_torch_gdn(patch, bridge_gdn)

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

    if native_torch_gdn is None:
        logger.warning(
            'FLA GDN is unavailable and no native Torch fallback was found; '
            'keep the current Megatron GDN implementation: %s', fla_error)
        return

    native_torch_gdn = _adapt_native_torch_gdn(native_torch_gdn)
    try:
        _apply_gdn_patch(MindSpeedPatchesManager, patch, native_torch_gdn)
    except Exception as e:
        logger.warning(
            'FLA GDN is unavailable and native Torch fallback setup failed; '
            'keep the current Megatron GDN implementation: %s', e)
        return

    global _LOGGED_TORCH_GDN_FALLBACK
    if not _LOGGED_TORCH_GDN_FALLBACK:
        logger.warning(
            'FLA GDN is unavailable (%s); falling back to native Torch chunk_gated_delta_rule. '
            'Packed Megatron GDN still requires flash-linear-attention.',
            fla_error,
        )
        _LOGGED_TORCH_GDN_FALLBACK = True


def patch_mindspeed_fla_gdn_implementation() -> None:
    """Best-effort preference for upstream FLA with a native Torch fallback."""
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

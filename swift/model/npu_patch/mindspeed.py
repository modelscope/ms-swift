# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib
import inspect
import sys
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()

_ORIGINAL_MINDSPEED_TE_CP_CLASS = None
_FLA_GDN_PATCH_TARGET = 'fla.ops.gated_delta_rule.chunk_gated_delta_rule'


def patch_mindspeed_fla_gdn_implementation() -> None:
    """Keep Megatron GDN on upstream FLA after MindSpeed patch/repatch."""
    try:
        from mindspeed.patch_utils import MindSpeedPatchesManager
    except ImportError as e:
        raise RuntimeError('Failed to import the upstream FLA GDN implementation for Megatron.') from e

    patch = MindSpeedPatchesManager.patches_info.get(_FLA_GDN_PATCH_TARGET)
    if patch is not None and patch.orig_func is not None:
        # MindSpeed propagates its replacement into already imported submodules,
        # so importing from ``fla.ops.gated_delta_rule.chunk`` again is not enough.
        fla_chunk_gated_delta_rule = patch.orig_func
    else:
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
    if not fla_chunk_gated_delta_rule.__module__.startswith('fla.'):
        raise RuntimeError('Failed to locate the upstream FLA GDN implementation: '
                           f'resolved {fla_chunk_gated_delta_rule.__module__}.{fla_chunk_gated_delta_rule.__name__}.')

    if patch is not None:
        MindSpeedPatchesManager.register_patch(
            _FLA_GDN_PATCH_TARGET,
            fla_chunk_gated_delta_rule,
            force_patch=True,
        )
        MindSpeedPatchesManager.apply_patches()
    else:
        fla_gated_delta_rule = importlib.import_module('fla.ops.gated_delta_rule')
        fla_gated_delta_rule.chunk_gated_delta_rule = fla_chunk_gated_delta_rule

    # mcore-bridge may have cached the MindSpeed callable before a runtime repatch.
    bridge_gdn = sys.modules.get('mcore_bridge.model.modules.gated_delta_net')
    if bridge_gdn is not None:
        bridge_gdn.chunk_gated_delta_rule = fla_chunk_gated_delta_rule

    fla_gated_delta_rule = importlib.import_module('fla.ops.gated_delta_rule')
    resolved = fla_gated_delta_rule.chunk_gated_delta_rule
    if resolved is not fla_chunk_gated_delta_rule or not resolved.__module__.startswith('fla.'):
        raise RuntimeError(f'Failed to restore upstream FLA GDN: resolved {resolved.__module__}.{resolved.__name__}.')
    if bridge_gdn is not None and bridge_gdn.chunk_gated_delta_rule is not fla_chunk_gated_delta_rule:
        raise RuntimeError('Failed to restore the mcore-bridge cached upstream FLA GDN callable.')

    logger.info(
        'Using upstream FLA chunk_gated_delta_rule for Megatron GDN: module=%s, source=%s.',
        resolved.__module__,
        inspect.getsourcefile(inspect.unwrap(resolved)),
    )


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

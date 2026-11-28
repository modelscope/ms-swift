# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU non-fused MoE permutation optimization."""

import inspect
import torch

_ORIGINAL_PERMUTE = None


def permute(tokens,
            routing_map,
            probs=None,
            num_out_tokens=None,
            fused=False,
            drop_and_pad=False,
            tokens_per_expert=None,
            align_size=0):
    """Avoid the NPU bool-sort fallback while preserving stable token order."""
    if tokens.device.type != 'npu' or fused or drop_and_pad or align_size:
        return _ORIGINAL_PERMUTE(tokens, routing_map, probs, num_out_tokens, fused, drop_and_pad, tokens_per_expert,
                                 align_size)
    assert num_out_tokens is not None, 'num_out_tokens is required for permutation'
    # FP32 represents the boolean routing mask exactly.
    flat_map = routing_map.bool().T.contiguous().reshape(-1)
    flat_sorted = flat_map.float().argsort(descending=True, stable=True)[:num_out_tokens]
    sorted_indices = flat_sorted % tokens.shape[0]
    permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted] if probs is not None else None
    return tokens.index_select(0, sorted_indices), permuted_probs, sorted_indices, None, tokens_per_expert


def patch_megatron_moe():
    """Replace MCore permutation exports and cached dispatcher references."""
    global _ORIGINAL_PERMUTE
    if _ORIGINAL_PERMUTE is not None:
        return
    from megatron.core.transformer.moe import moe_utils, token_dispatcher

    if tuple(inspect.signature(moe_utils.permute).parameters) != tuple(inspect.signature(permute).parameters):
        raise RuntimeError('Unsupported MCore permute signature for the Swift NPU performance patch.')
    _ORIGINAL_PERMUTE = moe_utils.permute
    moe_utils.permute = token_dispatcher.permute = permute

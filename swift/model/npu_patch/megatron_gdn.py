# Copyright (c) ModelScope Contributors. All rights reserved.
"""Honor Q/K normalization in MegatronAdaptor's packed GDN fallback."""

import inspect
import torch.nn.functional as F

_ORIGINAL_CHUNK_GATED_DELTA_RULE = None


def chunk_gated_delta_rule(query,
                           key,
                           value,
                           g,
                           beta,
                           scale=None,
                           chunk_size=64,
                           initial_state=None,
                           output_final_state=False,
                           use_qk_l2norm_in_kernel=False,
                           cu_seqlens=None,
                           head_first=False):
    if cu_seqlens is not None and use_qk_l2norm_in_kernel:
        query = F.normalize(query.float(), dim=-1, eps=1e-6).to(query.dtype)
        key = F.normalize(key.float(), dim=-1, eps=1e-6).to(key.dtype)
        use_qk_l2norm_in_kernel = False
    return _ORIGINAL_CHUNK_GATED_DELTA_RULE(
        query,
        key,
        value,
        g,
        beta,
        scale=scale,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        head_first=head_first)


def patch_megatron_gdn():
    global _ORIGINAL_CHUNK_GATED_DELTA_RULE
    if _ORIGINAL_CHUNK_GATED_DELTA_RULE is not None:
        return
    from megatron_adaptor.patches.megatron import gdn_packed_seq
    from megatron_adaptor.patches.patch_manager import PatchesManager

    original = gdn_packed_seq._npu_chunk_gated_delta_rule
    if tuple(inspect.signature(original).parameters) != tuple(inspect.signature(chunk_gated_delta_rule).parameters):
        raise RuntimeError('Unsupported MegatronAdaptor GDN signature for the Swift NPU patch.')
    _ORIGINAL_CHUNK_GATED_DELTA_RULE = original
    # Refresh direct fallback imports as well as the optional FLA dummy export.
    target = 'megatron_adaptor.patches.megatron.gdn_packed_seq._npu_chunk_gated_delta_rule'
    PatchesManager.register_patch(target, chunk_gated_delta_rule)
    PatchesManager.patches_info[target].apply_patch()
    target = 'fla.ops.gated_delta_rule.chunk_gated_delta_rule'
    patch = PatchesManager.patches_info.get(target)
    if patch is not None and patch.patch_func is original:
        PatchesManager.register_patch(target, chunk_gated_delta_rule, force_patch=True)
        patch.apply_patch()
        # Bridge imports convolution from a newer FLA path than Adaptor's dummy export.
        from mcore_bridge.model.mm_gpts import qwen3_5
        if qwen3_5._fla_causal_conv1d is None:
            qwen3_5._fla_causal_conv1d = gdn_packed_seq._npu_causal_conv1d

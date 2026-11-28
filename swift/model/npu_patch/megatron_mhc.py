# Copyright (c) ModelScope Contributors. All rights reserved.
"""CANN mHC pre/post fusion for the GLM-5.3-Flash Megatron hybrid stack."""
import torch
from functools import partial, wraps

from . import _is_model_patch_enabled_from_argv
from .utils import get_cann_mhc_ops, import_optional_module, supports_cann_mhc


def _sequence_first(tensor):
    # Transpose backward produces a strided [B,S,...] gradient when B and S both exceed one.
    # Supply contiguous gradients to the CANN backward kernels.
    if tensor.requires_grad:
        tensor.register_hook(torch.Tensor.contiguous)
    return tensor.transpose(0, 1).contiguous()


def _cann_pre(module, native, op, hidden_states):
    if not supports_cann_mhc(hidden_states, module.n, module.hidden_size):
        return native(hidden_states)

    # MCore packs streams into [S,B,N*H]. The CANN wrapper consumes [B,S,N,H].
    streams = hidden_states.unflatten(-1, (module.n, module.hidden_size)).transpose(0, 1).contiguous()
    alpha = torch.cat((module.alpha_pre, module.alpha_post, module.alpha_res)).float()
    aggregated, h_post, h_res = op(streams, module.mapping_proj.weight.float(), alpha, module.bias.float(), module.n,
                                   module.sinkhorn_iterations, module.sinkhorn_eps, module.norm_eps)
    h_res = h_res.unflatten(-1, (module.n, module.n))
    return _sequence_first(aggregated), _sequence_first(h_res), _sequence_first(h_post), hidden_states


def _cann_post(native, op, h_res, residual, h_post, output, bias):
    if bias is not None or output.dtype != residual.dtype or not supports_cann_mhc(residual, *residual.shape[-2:]):
        return native(h_res, residual, h_post, output, bias)
    merged = op(
        residual.transpose(0, 1).contiguous(),
        h_res.transpose(0, 1).contiguous(),
        output.transpose(0, 1).contiguous(),
        h_post.transpose(0, 1).contiguous())
    return _sequence_first(merged)


def patch_megatron_mhc():
    """Choose the mHC operators once, when a compatible GLM layer is constructed."""
    if not _is_model_patch_enabled_from_argv():
        return
    module = import_optional_module('megatron.core.transformer.hyper_connection')
    HyperConnectionModule = getattr(module, 'HyperConnectionModule', None)
    if not hasattr(HyperConnectionModule, '_forward_normal'):
        return

    original_init = HyperConnectionModule.__init__
    if getattr(original_init, '_glm53_cann_mhc', False):
        return

    @wraps(original_init)
    def init(self, config, layer_number):
        original_init(self, config, layer_number)
        if getattr(config, 'hf_model_type', None) != 'glm5_next':
            return
        if not (self.n == 4 and self.sinkhorn_iterations == 20 and getattr(config, 'mhc_norm_eps_inside_sqrt', False)
                and getattr(config, 'mhc_keep_mappings_in_fp32', False)
                and self.sinkhorn_eps == getattr(self, 'compute_h_eps', None)):
            return
        ops = get_cann_mhc_ops()
        if ops is not None:
            pre, post = ops
            self._forward_normal = partial(_cann_pre, self, self._forward_normal, pre)
            self._h_post_bda_op = partial(_cann_post, self._h_post_bda_op, post)

    init._glm53_cann_mhc = True
    HyperConnectionModule.__init__ = init

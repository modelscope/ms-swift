# Copyright (c) ModelScope Contributors. All rights reserved.
"""CANN mHC pre/post fusion for the GLM-5.3-Flash Megatron hybrid stack."""
import torch
from functools import partial, wraps

from . import _is_model_patch_enabled_from_argv
from .mhc import get_mhc_mode, get_mhc_ops, supports_auto_mhc
from .utils import import_optional_module


def _sequence_first(tensor):
    # Transpose backward produces a strided [B,S,...] gradient when B and S both exceed one.
    # Supply contiguous gradients to the CANN backward kernels.
    if tensor.requires_grad:
        tensor.register_hook(torch.Tensor.contiguous)
    return tensor.transpose(0, 1).contiguous()


def _cann_pre(module, native, op, hidden_states, *, automatic=False):
    if automatic and not supports_auto_mhc(hidden_states, module.n, module.hidden_size):
        return native(hidden_states)
    if hidden_states.device.type != 'npu' or hidden_states.dtype != torch.bfloat16:
        return native(hidden_states)

    # MCore packs streams into [S,B,N*H]. The CANN wrapper consumes [B,S,N,H].
    streams = hidden_states.unflatten(-1, (module.n, module.hidden_size)).transpose(0, 1).contiguous()
    alpha = torch.cat((module.alpha_pre, module.alpha_post, module.alpha_res)).float()
    aggregated, h_post, h_res = op(streams, module.mapping_proj.weight.float(), alpha, module.bias.float(), module.n,
                                   module.sinkhorn_iterations, module.sinkhorn_eps, module.norm_eps)
    h_res = h_res.unflatten(-1, (module.n, module.n))
    return _sequence_first(aggregated), _sequence_first(h_res), _sequence_first(h_post), hidden_states


def _cann_post(native, op, h_res, residual, h_post, output, bias, *, automatic=False):
    if automatic and not supports_auto_mhc(residual, *residual.shape[-2:]):
        return native(h_res, residual, h_post, output, bias)
    if (residual.device.type != 'npu' or residual.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or bias is not None):
        return native(h_res, residual, h_post, output, bias)
    merged = op(
        residual.transpose(0, 1).contiguous(),
        h_res.transpose(0, 1).contiguous(),
        output.transpose(0, 1).contiguous(),
        h_post.transpose(0, 1).contiguous())
    return _sequence_first(merged)


def patch_megatron_mhc(mode=None):
    """Choose the mHC operators once, when a compatible GLM layer is constructed."""
    if not _is_model_patch_enabled_from_argv():
        return
    # Older Python callers passed 'cann' to install both operators.
    mode = get_mhc_mode('both' if mode == 'cann' else mode)
    if mode == 'off':
        return

    automatic = mode == 'auto'
    if automatic:
        module = import_optional_module('megatron.core.transformer.hyper_connection')
        HyperConnectionModule = getattr(module, 'HyperConnectionModule', None)
        if not hasattr(HyperConnectionModule, '_forward_normal'):
            return
    else:
        from megatron.core.transformer.hyper_connection import HyperConnectionModule

    from swift.utils import get_logger

    original_init = HyperConnectionModule.__init__
    if getattr(original_init, '_glm53_cann_mhc', False):
        return

    @wraps(original_init)
    def init(self, config, layer_number):
        original_init(self, config, layer_number)
        if getattr(config, 'hf_model_type', None) != 'glm5_next':
            return
        if not (self.n == 4 and getattr(config, 'mhc_norm_eps_inside_sqrt', False)
                and getattr(config, 'mhc_keep_mappings_in_fp32', False)
                and self.sinkhorn_eps == getattr(self, 'compute_h_eps', None)):
            if automatic:
                return
            raise ValueError('CANN GLM mHC requires four streams, FP32 mixing, epsilon inside sqrt '
                             'and matching gate/Sinkhorn epsilons')
        if automatic and self.sinkhorn_iterations != 20:
            return
        (pre, _), (post, _) = get_mhc_ops(mode)
        if pre is not None:
            self._forward_normal = partial(_cann_pre, self, self._forward_normal, pre, automatic=automatic)
        if post is not None:
            self._h_post_bda_op = partial(_cann_post, self._h_post_bda_op, post, automatic=automatic)
        if pre is not None or post is not None:
            stages = 'pre/post' if pre and post else mode
            get_logger().info_once(f'Enabled CANN mHC {stages} for Megatron GLM HyperConnectionModule.')

    init._glm53_cann_mhc = True
    HyperConnectionModule.__init__ = init

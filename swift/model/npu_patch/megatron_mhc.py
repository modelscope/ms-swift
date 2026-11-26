# Copyright (c) ModelScope Contributors. All rights reserved.
"""CANN mHC pre/post fusion for the GLM-5.3-Flash Megatron hybrid stack."""
import os
import torch
from functools import cache, partial, wraps


def _sequence_first(tensor):
    # Transpose backward produces a strided [B,S,...] gradient when B and S both exceed one.
    # Supply contiguous gradients to the CANN backward kernels.
    if tensor.requires_grad:
        tensor.register_hook(torch.Tensor.contiguous)
    return tensor.transpose(0, 1).contiguous()


def _cann_pre(module, native, op, hidden_states):
    if hidden_states.device.type != 'npu' or hidden_states.dtype != torch.bfloat16:
        return native(hidden_states)

    # MCore packs streams into [S,B,N*H]. The CANN wrapper consumes [B,S,N,H].
    streams = hidden_states.unflatten(-1, (module.n, module.hidden_size)).transpose(0, 1).contiguous()
    alpha = torch.cat((module.alpha_pre, module.alpha_post, module.alpha_res)).float()
    aggregated, h_post, h_res = op(streams, module.mapping_proj.weight.float(), alpha, module.bias.float(), module.n,
                                   module.sinkhorn_iterations, module.sinkhorn_eps, module.norm_eps)
    h_res = h_res.unflatten(-1, (module.n, module.n))
    return _sequence_first(aggregated), _sequence_first(h_res), _sequence_first(h_post), hidden_states


def _cann_post(native, op, h_res, residual, h_post, output, bias):
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
    """Install once per process. MCore 0.18 and the default mode need no CANN mHC imports."""
    if mode is None:
        mode = os.environ.get('GLM_MEGATRON_MHC', 'off')
    if mode == 'off':
        return
    if mode != 'cann':
        raise ValueError('GLM_MEGATRON_MHC must be off or cann')

    from megatron.core.transformer.hyper_connection import HyperConnectionModule

    from swift.utils import get_logger
    from .glm5_next_mhc import cann_package, resolve_op

    original_init = HyperConnectionModule.__init__
    if getattr(original_init, '_glm53_cann_mhc', False):
        return

    @cache
    def get_cann_ops():
        package = cann_package()
        pre, _ = resolve_op(package, 'mhc_pre_sinkhorn', 'cann')
        post, _ = resolve_op(package, 'mhc_post', 'cann')
        return pre, post

    @wraps(original_init)
    def init(self, config, layer_number):
        original_init(self, config, layer_number)
        if getattr(config, 'hf_model_type', None) != 'glm5_next':
            return
        if not (self.n == 4 and config.mhc_norm_eps_inside_sqrt and config.mhc_keep_mappings_in_fp32
                and self.sinkhorn_eps == self.compute_h_eps):
            raise ValueError('CANN GLM mHC requires four streams, FP32 mixing, epsilon inside sqrt '
                             'and matching gate/Sinkhorn epsilons')
        pre, post = get_cann_ops()
        self._forward_normal = partial(_cann_pre, self, self._forward_normal, pre)
        self._h_post_bda_op = partial(_cann_post, self._h_post_bda_op, post)
        get_logger().info_once('Enabled CANN mHC pre/post for Megatron GLM HyperConnectionModule.')

    init._glm53_cann_mhc = True
    HyperConnectionModule.__init__ = init

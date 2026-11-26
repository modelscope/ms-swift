# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional mHC forward and backward fusion for the GLM-5.3-Flash text decoder."""
import inspect
import os
import torch
from functools import wraps

from swift.utils.logger import get_logger
from .mhc import get_mhc_ops, supports_auto_mhc

logger = get_logger()


def native_post(residual, comb, h_out, post):
    # Preserve the HF rounding points for the reference comparison.
    dtype = residual.dtype
    mixed = comb.to(dtype).transpose(-1, -2) @ residual
    return post.to(dtype).unsqueeze(-1) * h_out.unsqueeze(-2) + mixed


def with_fused_post(original, merge):
    # Transformers inlines both residual merges in this method.
    def decoder_forward(self,
                        hidden_states,
                        attention_mask=None,
                        position_ids=None,
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=None,
                        prev_topk_indices=None,
                        **kwargs):
        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        topk_indices = None
        if self.block_type == 'linear_attention':
            hidden_states = self.self_attn(
                hidden_states=hidden_states, cache_params=past_key_values, attention_mask=attention_mask, **kwargs)
        else:
            hidden_states, _, topk_indices = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                prev_topk_indices=prev_topk_indices,
                **kwargs)
        residual = merge(residual, comb, hidden_states, post)
        post, comb, hidden_states = self.ffn_hc(residual)
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return merge(residual, comb, hidden_states, post), topk_indices

    if inspect.unwrap(original) is not original or original.__closure__:
        raise RuntimeError('GLM mHC post requires an unwrapped Transformers decoder forward. Use GLM_MHC=pre.')
    params = inspect.signature(original).parameters.values()
    native_params = [param.replace(annotation=inspect.Parameter.empty) for param in params]
    if native_params != list(inspect.signature(decoder_forward).parameters.values()):
        raise RuntimeError('Unsupported GLM-5.3 decoder forward signature for mHC post. Use GLM_MHC=pre.')
    return wraps(original)(decoder_forward)


def make_pre(cann_pre, original, backend, automatic=False):
    # CANN releases expose either an autograd wrapper with eight arguments or
    # a wrapper with an explicit backward-intermediates flag.
    sig = inspect.signature(cann_pre)
    flags = {'outflag', 'needbackward'}
    flag = next((name for name in sig.parameters if name.replace('_', '').lower() in flags), None)
    if flag:
        sig.bind(*([None] * 8), **{flag: True})
    else:
        sig.bind(*([None] * 8))

    @wraps(original)
    def fused(self, hidden_streams):
        x = hidden_streams
        if automatic and (self.hc_sinkhorn_iters != 20 or not supports_auto_mhc(x, self.hc_mult, x.shape[-1])):
            return original(self, x)
        if x.device.type != 'npu' or x.dtype != torch.bfloat16:
            if backend == 'cann':
                raise ValueError('CANN mHC pre requires BF16 hidden streams on NPU')
            return original(self, x)
        batch, seq, streams, _ = x.shape
        phi, alpha, bias = self.fn.float(), self.scale.float(), self.base.float()
        kwargs = {}
        if flag:
            kwargs[flag] = torch.is_grad_enabled() and any(t.requires_grad for t in (x, phi, alpha, bias))
        values = cann_pre(x.contiguous(), phi, alpha, bias, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps,
                          self.input_norm.eps, **kwargs)
        h_in, post, comb = values[:3]
        return post, comb.reshape(batch, seq, streams, streams), h_in

    return fused


def selected_backend():
    backend = os.environ.get('GLM_MHC_BACKEND', 'cann').lower()
    if backend not in ('auto', 'cann'):
        raise ValueError('GLM_MHC_BACKEND must be auto or cann')
    return backend


def make_forwards(mode, original, backend):
    (pre_op, pre_backend), (mhc_post, post_backend) = get_mhc_ops(mode, backend)
    pre = original
    if pre_op is not None:
        pre = make_pre(pre_op, original, backend, automatic=mode == 'auto')
    if mhc_post is not None:

        def post(residual, comb, h_out, post_weight):
            if mode == 'auto' and (not supports_auto_mhc(residual, *residual.shape[-2:])
                                   or h_out.dtype != residual.dtype):
                return native_post(residual, comb, h_out, post_weight)
            if residual.device.type != 'npu' or residual.dtype != torch.bfloat16:
                if backend == 'cann':
                    raise ValueError('CANN mHC post requires BF16 residuals on NPU')
                return native_post(residual, comb, h_out, post_weight)
            return mhc_post(residual.contiguous(),
                            comb.float().contiguous(), h_out.contiguous(),
                            post_weight.float().contiguous())
    else:
        post = native_post

    return pre, post, pre_backend, post_backend


def get_glm5_next_mhc_patches(glm, mode):
    try:
        backend = 'cann' if mode == 'auto' else selected_backend()
        original = glm.Glm5NextTextHyperConnection.forward
        pre, post, pre_name, post_name = make_forwards(mode, original, backend)
        patches = {}
        if pre is not original:
            patches['Glm5NextTextHyperConnection.forward'] = pre
        if post is not native_post:
            patches['Glm5NextTextDecoderLayer.forward'] = with_fused_post(glm.Glm5NextTextDecoderLayer.forward, post)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        if mode != 'auto':
            raise
        logger.debug('Using native Transformers GLM mHC: %s', exc)
        return {}
    if patches:
        logger.info(f'[glm-mhc] mode={mode} pre={pre_name} post={post_name}')
    return patches

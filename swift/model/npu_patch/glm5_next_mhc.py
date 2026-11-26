# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional mHC forward and backward fusion for the GLM-5.3-Flash text decoder."""
import inspect
from functools import wraps

from swift.utils.logger import get_logger
from .utils import get_cann_mhc_ops, supports_cann_mhc

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
        return None
    params = inspect.signature(original).parameters.values()
    native_params = [param.replace(annotation=inspect.Parameter.empty) for param in params]
    if native_params != list(inspect.signature(decoder_forward).parameters.values()):
        return None
    return wraps(original)(decoder_forward)


def make_pre(cann_pre, original):

    @wraps(original)
    def fused(self, hidden_streams):
        x = hidden_streams
        if self.hc_sinkhorn_iters != 20 or not supports_cann_mhc(x, self.hc_mult, x.shape[-1]):
            return original(self, x)
        h_in, post, comb = cann_pre(x.contiguous(), self.fn.float(), self.scale.float(), self.base.float(),
                                    self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps, self.input_norm.eps)
        return post, comb.unflatten(-1, (self.hc_mult, self.hc_mult)), h_in

    return fused


def make_post(cann_post):

    def fused(residual, comb, h_out, post):
        if h_out.dtype != residual.dtype or not supports_cann_mhc(residual, *residual.shape[-2:]):
            return native_post(residual, comb, h_out, post)
        return cann_post(residual.contiguous(),
                         comb.float().contiguous(), h_out.contiguous(),
                         post.float().contiguous())

    return fused


def get_glm5_next_mhc_patches(glm):
    ops = get_cann_mhc_ops()
    hyper_connection = getattr(glm, 'Glm5NextTextHyperConnection', None)
    decoder = getattr(glm, 'Glm5NextTextDecoderLayer', None)
    if ops is None or hyper_connection is None or decoder is None:
        return {}
    pre, post = ops
    forward = with_fused_post(decoder.forward, make_post(post))
    if forward is None:
        logger.debug('Skipping CANN mHC for an unsupported GLM-5.3 decoder forward.')
        return {}
    return {
        'Glm5NextTextHyperConnection.forward': make_pre(pre, hyper_connection.forward),
        'Glm5NextTextDecoderLayer.forward': forward
    }

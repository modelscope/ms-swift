# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from typing import Optional

try:
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet as _GatedDeltaNet
    from megatron.core.ssm.gated_delta_net import (causal_conv1d_fn, chunk_gated_delta_rule, l2norm,
                                                   torch_chunk_gated_delta_rule)
except ImportError:
    _GatedDeltaNet = object


class GatedDeltaNet(_GatedDeltaNet):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.

        """
        # TODO: Deal with attention_mask
        from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()), 'GDN does not currently support dynamic inference batching.'
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError('GDN does not support inference for now.')

        if packed_seq_params is not None:
            # TODO: support packed sequence
            raise NotImplementedError('GDN does not support packed sequence for now.')

        # Input projection
        nvtx_range_push(suffix='in_proj')
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix='in_proj')

        # Transpose: s b x --> b s x
        # From sbhd to bshd format
        qkvzba = qkvzba.transpose(0, 1)

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha
        num_key_heads_per_device = self.num_key_heads // self.tp_size
        qkvzba = qkvzba.view(qkvzba.shape[:-1]
                             + (num_key_heads_per_device, qkvzba.shape[-1] // num_key_heads_per_device))
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim * 2 + self.v_dim) // self.num_key_heads,
                self.v_dim // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
                self.num_value_heads // self.num_key_heads,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)
        qkv = qkv.reshape(batch, seq_len, -1)

        # Convolution on qkv
        qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
        nvtx_range_push(suffix='conv1d')
        if (causal_conv1d_fn is None) or self.config.deterministic_mode:
            qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        else:
            assert self.activation in ['silu', 'swish']
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1),  # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        nvtx_range_pop(suffix='conv1d')
        # Split qkv into query, key, and value
        qkv = qkv.transpose(1, 2)  # b, d, s -> b, s, d
        qkv = qkv.view(qkv.shape[:-1] + (num_key_heads_per_device, qkv.shape[-1] // num_key_heads_per_device))
        query, key, value = torch.split(
            qkv,
            [self.qk_dim // self.num_key_heads, self.qk_dim // self.num_key_heads, self.v_dim // self.num_key_heads],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        # Apply L2 norm to query and key
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix='g_and_beta')
        g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias)  # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix='g_and_beta')

        nvtx_range_push(suffix='gated_delta_rule')
        if self.config.deterministic_mode:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        nvtx_range_pop(suffix='gated_delta_rule')

        # RMSNorm
        nvtx_range_push(suffix='gated_norm')
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix='gated_norm')

        # Transpose: b s x --> s b x
        # From bshd back to sbhd format
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        # Output projection
        nvtx_range_push(suffix='out_proj')
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix='out_proj')

        return out, out_bias

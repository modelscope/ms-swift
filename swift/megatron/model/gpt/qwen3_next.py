# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import deprecate_inference_params, is_fa_min_version, nvtx_range_pop, nvtx_range_push
from megatron.training import get_args
from torch import nn

from swift.llm import ModelType
from swift.utils import get_logger
from ..constant import MegatronModelType
from ..gpt_model import GPTModel
from ..register import MegatronModelMeta, register_megatron_model
from .config import convert_gpt_hf_config

try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward
    from flashattn_hopper.flash_attn_interface import flash_attn_with_kvcache as flash_attn3_with_kvcache

    HAVE_FA3 = True
except Exception:
    HAVE_FA3 = False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet as _Qwen3NextGatedDeltaNet
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDynamicCache, Qwen3NextConfig
except ImportError:
    _Qwen3NextGatedDeltaNet = None
    Qwen3NextDynamicCache = None
    Qwen3NextConfig = None

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None

logger = get_logger()


class Qwen3NextRMSNormGated(MegatronModule):
    # code borrowed from huggingface/transformers
    def __init__(self, config, hidden_size, **kwargs):
        super().__init__(config)
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(self.hidden_size))
        self.variance_epsilon = config.layernorm_epsilon

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


class Qwen3NextSelfAttention(SelfAttention):

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, *args, **kwargs):
        super(SelfAttention, self).__init__(config, submodules, *args, attention_type='self', **kwargs)
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            2 * self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.model_comm_pgs.tp,
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    # Code borrowed from NVIDIA/Megatron-LM
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False)
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version(
                '2.7.3'), 'flash attn verion v2.7.3 and above is required for dynamic batching.'

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, ) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix='qkv')
        query, key, value, gate = self.get_query_key_value_tensors(hidden_states, key_value_states)
        nvtx_range_pop(suffix='qkv')

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (inference_context is not None and inference_context.is_decode_only() and not self.training)

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix='adjust_key_value')
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[self.layer_number]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        if (in_decode_mode and self.config.enable_cuda_graph and inference_context.is_static_batching()):
            raise ValueError('CUDA graphs must use flash decode with static batching!')

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            ))

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix='adjust_key_value')

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix='rotary_pos_emb')
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        cp_group=self.model_comm_pgs.cp,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(query, q_pos_emb, self.config, cu_seqlens_q,
                                                                     self.model_comm_pgs.cp)
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    cp_group=self.model_comm_pgs.cp,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix='rotary_pos_emb')

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix='core_attention')
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, kv_lengths_decode_only, max_seqlen_k = (inference_context.cu_kv_lengths())

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    kv_lengths_decode_only,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix='core_attention')

        # =================
        # Output. [sq, b, h]
        # =================

        core_attn_out = core_attn_out * torch.sigmoid(gate.reshape_as(core_attn_out))
        nvtx_range_push(suffix='linear_proj')
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix='linear_proj')

        return output, bias

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition * 2 + 2)
             * self.hidden_size_per_attention_head),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition
             * self.hidden_size_per_attention_head * 2),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        query, gate = query[:, :, ::2], query[:, :, 1::2]
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, gate


if _Qwen3NextGatedDeltaNet is None:
    Qwen3NextGatedDeltaNet = None
else:

    class Qwen3NextGatedDeltaNet(_Qwen3NextGatedDeltaNet, MegatronModule):
        # Code borrowed from huggingface/transformers
        def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
            args = get_args()
            from transformers.models.qwen3_next.modeling_qwen3_next import (
                FusedRMSNormGated,
                causal_conv1d_fn,
                causal_conv1d_update,
                torch_causal_conv1d_update,
                chunk_gated_delta_rule,
                torch_chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
                torch_recurrent_gated_delta_rule,
                is_fast_path_available,
            )
            MegatronModule.__init__(self, config)
            self.model_comm_pgs = kwargs['model_comm_pgs']
            self.hidden_size = config.hidden_size
            self.num_v_heads = args.linear_num_value_heads
            self.num_k_heads = args.linear_num_key_heads
            self.head_k_dim = args.linear_key_head_dim
            self.head_v_dim = args.linear_value_head_dim
            self.key_dim = self.head_k_dim * self.num_k_heads
            self.value_dim = self.head_v_dim * self.num_v_heads

            self.conv_kernel_size = args.linear_conv_kernel_dim
            self.layer_idx = layer_number  # not use during trainging
            self.activation = 'silu'
            self.act = nn.SiLU()
            self.layer_norm_epsilon = config.layernorm_epsilon

            # QKV
            self.conv_dim = self.key_dim * 2 + self.value_dim
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                bias=False,
                kernel_size=self.conv_kernel_size,
                groups=self.conv_dim,
                padding=self.conv_kernel_size - 1,
            )

            # projection of the input hidden states
            projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
            projection_size_ba = self.num_v_heads * 2
            self.in_proj_qkvz = build_module(
                submodules.linear_qkv,
                self.hidden_size,
                projection_size_qkvz,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkvz',
                tp_group=self.model_comm_pgs.tp,
            )
            self.in_proj_ba = build_module(
                submodules.linear_qkv,
                self.hidden_size,
                projection_size_ba,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='ba',
                tp_group=self.model_comm_pgs.tp,
            )

            # time step projection (discretization)
            # instantiate once and copy inv_dt in init_weights of PretrainedModel
            self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

            A = torch.empty(self.num_v_heads).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))

            self.norm = (
                Qwen3NextRMSNormGated(self.config, self.head_v_dim) if FusedRMSNormGated is None else FusedRMSNormGated(
                    self.head_v_dim,
                    eps=self.layer_norm_epsilon,
                    activation=self.activation,
                    device=torch.cuda.current_device(),
                    dtype=args.torch_dtype,
                ))
            self.out_proj = build_module(
                submodules.linear_proj,
                self.value_dim,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='proj',
                tp_group=self.model_comm_pgs.tp,
            )

            self.causal_conv1d_fn = causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
            self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
            self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

            if not is_fast_path_available:
                logger.warning_once(
                    'The fast path is not available because one of the required library is not installed. '
                    'Falling back to torch implementation. To install follow '
                    'https://github.com/fla-org/flash-linear-attention#installation'
                    ' and https://github.com/Dao-AILab/causal-conv1d')

        def forward(self, hidden_states: torch.Tensor, **kwargs):
            # Set up dimensions for reshapes later
            args = get_args()
            if args.padding_free:
                packed_seq_params = kwargs['packed_seq_params']
                new_hidden_states = hidden_states.new_zeros(
                    (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item(), hidden_states.shape[-1]))
                attention_mask = hidden_states.new_zeros(
                    (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item()), dtype=torch.bool)
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                for i in range(packed_seq_params.num_samples):
                    start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
                    attention_mask[i, :end - start] = True
                    new_hidden_states[i, :end - start] = hidden_states[start:end, 0]
                hidden_states = new_hidden_states
            else:
                hidden_states = hidden_states.transpose(0, 1)
                from transformers.models.qwen3_next.modeling_qwen3_next import apply_mask_to_padding_states
                attention_mask = kwargs['attention_mask'].sum(dim=(1, 3)) > 0
                hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

            projected_states_qkvz = self.in_proj_qkvz(hidden_states)[0]
            projected_states_ba = self.in_proj_ba(hidden_states)[0]
            query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
            query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

            mixed_qkv = torch.cat((query, key, value), dim=-1)
            mixed_qkv = mixed_qkv.transpose(1, 2)

            batch_size, seq_len, _ = hidden_states.shape
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(
                mixed_qkv,
                [
                    self.key_dim,
                    self.key_dim,
                    self.value_dim,
                ],
                dim=-1,
            )
            query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
            key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
            value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

            beta = b.sigmoid()
            # If the model is loaded in fp16, without the .float() here, A might be -inf
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            if self.num_v_heads // self.num_k_heads > 1:
                query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
                key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

            z_shape_og = z.shape
            # reshape input data into 2D tensor
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape_og)
            core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

            output, bias = self.out_proj(core_attn_out)
            if args.padding_free:
                output = output[attention_mask][:, None]
            else:
                output = output.transpose(0, 1)
            return output, bias


def get_local_layer_specs(config, layer_specs, vp_stage=None):
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id] for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage)
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset:offset + num_layers_to_build]
    return local_layer_specs


def get_qwen3_next_transformer_layer_spec(config):
    config.hetereogenous_dist_checkpoint = True
    args = get_args()
    layer_norm_impl = TENorm
    moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        use_kitchen=config.use_kitchen,
    )
    layer_specs = []
    for layer_type in args.layer_types:
        layer_spec = deepcopy(moe_layer_spec)
        if layer_type == 'linear_attention':
            layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
            layer_spec.submodules.input_layernorm = layer_norm_impl
            layer_spec.submodules.self_attention.module = Qwen3NextGatedDeltaNet
        elif layer_type == 'full_attention':
            layer_spec.submodules.self_attention.module = Qwen3NextSelfAttention
        layer_specs.append(layer_spec)

    local_layer_specs = get_local_layer_specs(config, layer_specs)

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)

    return block_spec


def convert_mcore2hf_qwen3_next(hf_model, mg_model):
    from .mcore2hf import set_mlp_state, set_attn_state
    args = get_args()
    hf_model.model.embed_tokens.weight.data.copy_(mg_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        hf_model.lm_head.weight.data.copy_(mg_model.output_layer.weight)
    hf_model.model.norm.weight.data.copy_(mg_model.decoder.final_layernorm.weight - 1)
    for layer_idx in range(args.num_layers):
        layer_type = args.layer_types[layer_idx]
        mg_layer = mg_model.decoder.layers[layer_idx]
        hf_layer = hf_model.model.layers[layer_idx]
        mg_attn = mg_layer.self_attention

        if layer_type == 'linear_attention':
            hf_layer.linear_attn.load_state_dict(mg_attn.state_dict(), strict=False)
            hf_layer.input_layernorm.weight.data.copy_(mg_layer.input_layernorm.weight - 1)
        elif layer_type == 'full_attention':
            hf_attn = hf_layer.self_attn
            set_attn_state(args, mg_attn, hf_attn)
            hf_layer.input_layernorm.weight.data.copy_(mg_attn.linear_qkv.layer_norm_weight - 1)
            if args.qk_layernorm:
                hf_attn.q_norm.weight.data.copy_(mg_attn.q_layernorm.weight - 1)
                hf_attn.k_norm.weight.data.copy_(mg_attn.k_layernorm.weight - 1)

        set_mlp_state(args, mg_layer.mlp, hf_layer.mlp)
        hf_layer.post_attention_layernorm.weight.data.copy_(mg_layer.pre_mlp_layernorm.weight - 1)


def convert_hf2mcore_qwen3_next(hf_model, mg_model):
    from .hf2mcore import set_mlp_state, set_attn_state
    args = get_args()
    mg_model.embedding.word_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight + 1)
    for layer_idx in range(args.num_layers):
        layer_type = args.layer_types[layer_idx]
        mg_layer = mg_model.decoder.layers[layer_idx]
        hf_layer = hf_model.model.layers[layer_idx]
        mg_attn = mg_layer.self_attention

        if layer_type == 'linear_attention':
            mg_attn.load_state_dict(hf_layer.linear_attn.state_dict(), strict=False)
            mg_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight + 1)
        elif layer_type == 'full_attention':
            hf_attn = hf_layer.self_attn
            set_attn_state(args, mg_attn, hf_attn)
            mg_attn.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight + 1)
            if args.qk_layernorm:
                mg_attn.q_layernorm.weight.data.copy_(hf_attn.q_norm.weight + 1)
                mg_attn.k_layernorm.weight.data.copy_(hf_attn.k_norm.weight + 1)

        set_mlp_state(args, mg_layer.mlp, hf_layer.mlp)
        mg_layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight + 1)


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_next,
        [
            ModelType.qwen3_next,
            ModelType.qwen3_next_thinking,
        ],
        model_cls=GPTModel,
        convert_hf_config=convert_gpt_hf_config,
        get_transformer_layer_spec=get_qwen3_next_transformer_layer_spec,
        convert_mcore2hf=convert_mcore2hf_qwen3_next,
        convert_hf2mcore=convert_hf2mcore_qwen3_next,
    ))

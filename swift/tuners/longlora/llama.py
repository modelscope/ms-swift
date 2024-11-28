# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from dvlab-research/LongLoRA.

import math
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Cache, StaticCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from swift.utils import get_logger

logger = get_logger()


def _preprocess_qkv_fa2(attn_module, query_states, key_states, value_states, attention_mask):
    if attn_module.training:
        bsz, q_len = query_states.shape[:2]
        group_size = int(q_len * attn_module.config.group_size_ratio)
        if q_len % group_size != 0:
            raise ValueError(f'The sequence length {q_len} should'
                             f'be able to be splitted by the group_ratio {attn_module.config.group_size_ratio}')

        num_group = q_len // group_size

        def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
            qkv[:, :, num_heads // 2:] = qkv[:, :, num_heads // 2:].roll(-group_size // 2, dims=1)
            qkv = qkv.reshape(bsz * num_group, group_size, num_heads, head_dim)
            return qkv

        query_states = shift(query_states, bsz, q_len, group_size, attn_module.num_heads, attn_module.head_dim)
        key_states = shift(key_states, bsz, q_len, group_size, attn_module.num_heads, attn_module.head_dim)
        value_states = shift(value_states, bsz, q_len, group_size, attn_module.num_heads, attn_module.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :group_size].repeat(num_group, 1)

    return query_states, key_states, value_states, attention_mask


def _preprocess_qkv(attn_module, query_states, key_states, value_states, attention_mask):
    if attn_module.training:
        bsz, _, q_len = query_states.shape[:3]
        group_size = int(q_len * attn_module.config.group_size_ratio)
        if q_len % group_size != 0:
            raise ValueError(f'The sequence length {q_len} should'
                             f'be able to be splitted by the group_ratio {attn_module.config.group_size_ratio}')

        num_group = q_len // group_size

        def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
            qkv = qkv.transpose(1, 2)
            qkv = qkv.reshape(bsz * num_group, group_size, num_heads, head_dim)
            return qkv.transpose(1, 2)

        query_states = shift(query_states, bsz, q_len, group_size, attn_module.num_heads, attn_module.head_dim)
        key_states = shift(key_states, bsz, q_len, group_size, attn_module.num_heads, attn_module.head_dim)
        value_states = shift(value_states, bsz, q_len, group_size, attn_module.num_heads, attn_module.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)

    return query_states, key_states, value_states, attention_mask


def _postprocess_qkv(attn_module, attn_output, q_len):
    if attn_module.training:
        group_size = int(q_len * attn_module.config.group_size_ratio)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(-1, q_len, attn_module.num_heads, attn_module.head_dim)
        # shift back
        attn_output_clone = attn_output.clone()
        attn_output_clone[:, :, attn_module.num_heads // 2:] = attn_output[:, :, attn_module.num_heads // 2:].roll(
            group_size // 2, dims=1)
        attn_output = attn_output_clone
    return attn_output.transpose(1, 2)


def _postprocess_qkv_fa2(attn_module, attn_output, q_len):
    if attn_module.training:
        group_size = int(q_len * attn_module.config.group_size_ratio)
        attn_output = attn_output.reshape(-1, q_len, attn_module.num_heads, attn_module.head_dim)
        attn_output_clone = attn_output.clone()
        # shift back
        attn_output_clone[:, :, attn_module.num_heads // 2:] = attn_output[:, :, attn_module.num_heads // 2:].roll(
            group_size // 2, dims=1)
        attn_output = attn_output_clone
    return attn_output


# code borrowed from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py # noqa
def eager_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # patch position rolling
    query_states, key_states, value_states, causal_mask = _preprocess_qkv(self, query_states, key_states, value_states,
                                                                          attention_mask)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                         f' {attn_output.size()}')

    # patch position unrolling
    attn_output = _postprocess_qkv(self, attn_output, q_len)

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# code borrowed from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py # noqa
def fa2_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            '`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` '
            'make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers'
        )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout
    # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f'The input hidden states seems to be silently casted in float32, this might be related to'
            f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
            f' {target_dtype}.')

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

        # patch position rolling
        query_states, key_states, value_states, attention_mask = _preprocess_qkv_fa2(
            self, query_states, key_states, value_states, attention_mask)
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, 'sliding_window', None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    # patch position unrolling
    attn_output = _postprocess_qkv_fa2(self, attn_output, q_len)

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# code borrowed from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py  # noqa
def sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False

    # patch position rolling
    query_states, key_states, value_states, causal_mask = _preprocess_qkv(self, query_states, key_states, value_states,
                                                                          causal_mask)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    # patch position unrolling
    attn_output = _postprocess_qkv(self, attn_output, q_len)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def replace_llama_attn(model: nn.Module):
    layers = None
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList):
            layers = module
            break
    assert layers is not None
    for idx, m in enumerate(layers):
        if model.config._attn_implementation == 'flash_attention_2':
            cuda_major, cuda_minor = torch.cuda.get_device_capability()
            if cuda_major < 8:
                logger.warn(
                    'Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward.'  # noqa
                    'ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593')
            m.self_attn.forward = MethodType(fa2_forward, m.self_attn)
        elif model.config._attn_implementation == 'eager':
            m.self_attn.forward = MethodType(eager_forward, m.self_attn)
        elif model.config._attn_implementation == 'sdpa':
            m.self_attn.forward = MethodType(sdpa_forward, m.self_attn)

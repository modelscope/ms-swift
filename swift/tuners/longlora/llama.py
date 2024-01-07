# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from dvlab-research/LongLoRA.

import math
import warnings
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from torch import nn
from transformers.models.llama.modeling_llama import (apply_rotary_pos_emb,
                                                      repeat_kv, rotate_half)

from swift.utils import get_logger

logger = get_logger()


def forward_flashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func)
    from flash_attn.bert_padding import unpad_input, pad_input
    if not self.training:
        raise ValueError(
            'This function is only for training. For inference, please use forward_flashattn_inference.'
        )

    if output_attentions:
        warnings.warn(
            'Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.'
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                        self.head_dim).transpose(1, 2))
    key_states = (
        self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                        self.head_dim).transpose(1, 2))
    value_states = (
        self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                        self.head_dim).transpose(1, 2))
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states],
                      dim=2)  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = attention_mask.repeat(2, 1)
    nheads = qkv.shape[-2]
    # shift

    group_size = int(q_len * self.config.group_size_ratio)

    qkv = qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2,
                      self.head_dim).permute(0, 3, 1, 2, 4, 5).reshape(
                          bsz * 2, q_len, 3, self.num_heads // 2,
                          self.head_dim)

    x = rearrange(qkv, 'b s three h d -> b s (three h d)')
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    cu_q_len_tmp = torch.arange(
        0,
        max_s,
        group_size,
        device=key_padding_mask.device,
        dtype=cu_q_lens.dtype)
    cu_q_len_tmp2 = cu_q_len_tmp + group_size // 2
    cu_q_len_tmp2[cu_q_len_tmp2 >= max_s] = torch.iinfo(
        cu_q_len_tmp2.dtype).min
    cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp2]).repeat(
        bsz, 1) + cu_q_lens[:-1].unsqueeze(-1)
    cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)],
                          dim=-1).view(-1)
    cu_q_lens = cu_q_lens[cu_q_lens >= 0]
    x_unpad = rearrange(
        x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads // 2)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True)
    output = rearrange(
        pad_input(
            rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices, bsz * 2,
            q_len),
        'b s (h d) -> b s h d',
        h=nheads // 2,
    )
    output = output.reshape(bsz, 2, q_len, nheads // 2,
                            self.head_dim).transpose(1, 2).reshape(
                                bsz, q_len, nheads, self.head_dim)

    return self.o_proj(rearrange(output,
                                 'b s h d -> b s (h d)')), None, past_key_value


def forward_noflashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * self.config.group_size_ratio)

    if q_len % group_size != 0:
        raise ValueError(
            f'The sequence length {q_len} should'
            f'be able to be splitted by the group_ratio {self.config.group_size_ratio}'
        )

    num_group = q_len // group_size

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads
                             * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp,
            dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if self.training:
        # shift
        def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(
                -group_size // 2, dims=2)
            qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size),
                                              group_size, num_heads,
                                              head_dim).transpose(1, 2)
            return qkv

        query_states = shift(query_states, bsz, q_len, group_size,
                             self.num_heads, self.head_dim)
        key_states = shift(key_states, bsz, q_len, group_size, self.num_heads,
                           self.head_dim)
        value_states = shift(value_states, bsz, q_len, group_size,
                             self.num_heads, self.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :group_size, :
                                            group_size].repeat(
                                                num_group, 1, 1, 1)

    attn_weights = torch.matmul(query_states, key_states.transpose(
        2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads,
                                      self.head_dim)

    if self.training:
        # shift back
        attn_output[:, :, self.num_heads
                    // 2:] = attn_output[:, :, self.num_heads // 2:].roll(
                        group_size // 2, dims=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def forward_flashattn_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    from flash_attn import __version__ as flash_attn_version
    from flash_attn.flash_attn_interface import (
        flash_attn_func, flash_attn_varlen_kvpacked_func)
    from flash_attn.bert_padding import unpad_input, pad_input
    if output_attentions:
        warnings.warn(
            'Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.'
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, 'num_key_value_heads', self.num_heads)

    # shape: (b, s, num_heads, head_dim)
    q, k, v = (op(hidden_states).view(bsz, q_len, nh, self.head_dim)
               for op, nh in (
                   (self.q_proj, self.num_heads),
                   (self.k_proj, kv_heads),
                   (self.v_proj, kv_heads),
               ))  # noqa

    kv_seq_len = k.shape[1]
    if past_key_value is not None and len(past_key_value):
        past_kv_len = past_key_value.seen_tokens
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb(
        q.transpose(1, 2), k.transpose(1, 2), *cos_sin, position_ids)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    if use_cache:
        k, v = past_key_value.update(
            k.transpose(1, 2), v.transpose(1, 2), layer_idx=self.idx)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    else:
        past_key_value = None

    if attention_mask is None:
        output = flash_attn_func(
            q, k, v, 0.0, softmax_scale=None,
            causal=True).view(bsz, q_len, -1)
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:,
                                                                     -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask)
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                    inputs_embeds, past_key_values_length):
    if self.training:
        return attention_mask
    else:
        # [bsz, seq_len]
        if past_key_values_length > 0 and attention_mask is not None:
            attention_mask = torch.cat(
                (
                    torch.full(
                        (input_shape[0], past_key_values_length),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ),
                dim=-1,
            )

        if attention_mask is not None and torch.all(attention_mask):
            return None  # This uses the faster call when training with full samples

        return attention_mask


def forward_flashattn_inference_s2_attn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    if self.training:
        return forward_flashattn(self, hidden_states, attention_mask,
                                 position_ids, past_key_value,
                                 output_attentions, use_cache, padding_mask)
    else:
        return forward_flashattn_inference(self, hidden_states, attention_mask,
                                           position_ids, past_key_value,
                                           output_attentions, use_cache,
                                           padding_mask)


def patch_llama_forward(model: nn.Module, forward_function) -> None:
    # Compatible with transformers device_map
    for idx, m in enumerate(model.model.layers):
        new_forward = MethodType(forward_function, m.self_attn)
        if hasattr(model, '_old_forward'):
            m.self_attn._old_forward = new_forward
        else:
            m.self_attn.forward = new_forward
        m.self_attn.idx = idx


def replace_llama_attn(model: nn.Module, use_flash_attn=True):
    if use_flash_attn:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                'Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward.'
                'ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593'
            )
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
            _prepare_decoder_attention_mask)
        patch_llama_forward(model, forward_flashattn_inference_s2_attn)
    else:
        logger.warn(
            'The source code of LongLoRA without flash '
            'attention may has some problems, please use with careful.')
        patch_llama_forward(model, forward_noflashattn)

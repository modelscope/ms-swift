# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Derived from inclusionAI/Ling-3.0-tiny/modeling_bailing_moe_v3.py.
# Upstream source SHA256: c2509bf7ac580c262e2581d34d6403aa21682d2e10beb9ad85ad8820a7e33a40.
"""Portable projection/cache fixture derived from the frozen Ling3 MLA body.

Rotary is a no-op and attention uses CPU SDPA. The real remote class is tested
separately when official source is available.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.utils.deprecation import deprecate_kwarg


def apply_rotary_pos_emb_interleave(q, k, cos, sin):
    return q, k


def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, is_causal=attention_mask is None,
        scale=kwargs['scaling']).transpose(1, 2), None


@deprecate_kwarg('past_key_value', new_name='past_key_values', version='4.58')
def remote_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs]
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    if self.config.rope_interleave:
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        raise NotImplementedError('The fixture supports interleaved rotary only')
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    if past_key_values is not None:
        cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    if self.config._attn_implementation == 'flash_attention_2' and self.qk_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])
    attention_interface: Callable = eager_attention_forward
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs)
    if self.config._attn_implementation == 'flash_attention_2' and self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, :self.v_head_dim]
    if self.g_proj is not None:
        gate = self.g_proj(hidden_states)
        gate = F.sigmoid(gate.float()).type_as(hidden_states)
        if self.gated_attention_proj_granularity_type == 'head_wise':
            attn_output = attn_output * gate[:, :, :, None]
        else:
            attn_output = attn_output * gate.view(batch_size, seq_length, self.num_heads, self.v_head_dim)
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.dense(attn_output)
    return (attn_output, attn_weights, past_key_values)

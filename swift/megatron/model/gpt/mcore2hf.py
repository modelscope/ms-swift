# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

from megatron.training import get_args
from torch import nn


def set_mla_attn_state(args, mg_attn, hf_attn):
    hf_attn.o_proj.weight.data.copy_(mg_attn.linear_proj.weight)
    if args.q_lora_rank is None:
        hf_attn.q_proj.weight.data.copy_(mg_attn.linear_q_proj.weight)
    else:
        hf_attn.q_a_proj.weight.data.copy_(mg_attn.linear_q_down_proj.weight)
        hf_attn.q_b_proj.weight.data.copy_(mg_attn.linear_q_up_proj.weight)
    hf_attn.kv_a_proj_with_mqa.weight.data.copy_(mg_attn.linear_kv_down_proj.weight)
    hf_attn.kv_b_proj.weight.data.copy_(mg_attn.linear_kv_up_proj.weight)
    if args.qk_layernorm:
        hf_attn.kv_a_layernorm.weight.data.copy_(mg_attn.linear_kv_up_proj.layer_norm_weight)


def set_attn_state(args, mg_attn, hf_attn):
    num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads)
    # Copy weights
    mg_attn_weight = mg_attn.linear_qkv.weight.reshape((num_query_groups, -1, args.hidden_size))
    q_dim, kv_dim = hf_attn.q_proj.weight.shape[0] // num_query_groups, hf_attn.k_proj.weight.shape[
        0] // num_query_groups
    hf_attn.q_proj.weight.data.copy_(mg_attn_weight[:, :q_dim, :].reshape(-1, args.hidden_size))
    hf_attn.k_proj.weight.data.copy_(mg_attn_weight[:, q_dim:-kv_dim, :].reshape(-1, args.hidden_size))
    hf_attn.v_proj.weight.data.copy_(mg_attn_weight[:, -kv_dim:, :].reshape(-1, args.hidden_size))
    hf_attn.o_proj.weight.data.copy_(mg_attn.linear_proj.weight)

    # Copy bias
    if args.add_qkv_bias:
        mg_attn_bias = mg_attn.linear_qkv.bias.reshape((num_query_groups, -1))
        hf_attn.q_proj.bias.data.copy_(mg_attn_bias[:, :q_dim].reshape(-1))
        hf_attn.k_proj.bias.data.copy_(mg_attn_bias[:, q_dim:-kv_dim].reshape(-1))
        hf_attn.v_proj.bias.data.copy_(mg_attn_bias[:, -kv_dim:].reshape(-1))

    if args.qk_layernorm:
        q_norm = hf_attn.query_layernorm if hasattr(hf_attn, 'query_layernorm') else hf_attn.q_norm
        k_norm = hf_attn.key_layernorm if hasattr(hf_attn, 'key_layernorm') else hf_attn.k_norm
        q_norm.weight.data.copy_(mg_attn.q_layernorm.weight)
        k_norm.weight.data.copy_(mg_attn.k_layernorm.weight)


def _set_moe_state(args, mg_mlp, hf_mlp):
    hf_gate = hf_mlp.gate
    if hasattr(hf_gate, 'wg'):
        hf_gate = hf_gate.wg
    hf_gate.weight.data.copy_(mg_mlp.router.weight)
    if args.moe_router_enable_expert_bias:
        hf_gate.e_score_correction_bias.data.copy_(mg_mlp.router.expert_bias)
    if mg_mlp.shared_experts is not None:
        if hasattr(hf_mlp, 'shared_experts'):
            hf_shared_expert = hf_mlp.shared_experts
        elif hasattr(hf_mlp, 'shared_mlp'):
            hf_shared_expert = hf_mlp.shared_mlp
        else:
            hf_shared_expert = hf_mlp.shared_expert
        _set_mlp_state(mg_mlp.shared_experts, hf_shared_expert)
        if mg_mlp.shared_experts.gate_weight is not None:
            hf_mlp.shared_expert_gate.weight.data.copy_(mg_mlp.shared_experts.gate_weight)
    for expert_idx in range(args.num_experts):
        hf_expert = hf_mlp.experts
        if hasattr(hf_expert, '__len__'):
            hf_expert = hf_expert[expert_idx]
        _set_mlp_state(mg_mlp.experts, hf_expert, group_idx=expert_idx)


def _set_mlp_state(mg_mlp, hf_mlp, group_idx: Optional[int] = None):
    hf_grouped = not isinstance(hf_mlp.down_proj, nn.Module)
    if group_idx is None:
        linear_fc1_weight = mg_mlp.linear_fc1.weight
        linear_fc2_weight = mg_mlp.linear_fc2.weight
    else:
        linear_fc1_weight = getattr(mg_mlp.linear_fc1, f'weight{group_idx}')
        linear_fc2_weight = getattr(mg_mlp.linear_fc2, f'weight{group_idx}')

    if hf_grouped:
        hf_mlp.gate_up_proj.data[group_idx] = linear_fc1_weight.t()
        hf_mlp.down_proj.data[group_idx] = linear_fc2_weight.t()
    else:
        if hasattr(hf_mlp, 'gate_up_proj'):
            hf_mlp.gate_up_proj.weight.data.copy_(linear_fc1_weight)
        else:
            ffn_hidden_size = hf_mlp.gate_proj.weight.shape[0]
            hf_mlp.gate_proj.weight.data.copy_(linear_fc1_weight[:ffn_hidden_size])
            hf_mlp.up_proj.weight.data.copy_(linear_fc1_weight[ffn_hidden_size:])
        hf_mlp.down_proj.weight.data.copy_(linear_fc2_weight)


def set_mlp_state(args, mg_mlp, hf_mlp):
    if 'moe' in mg_mlp.__class__.__name__.lower():
        _set_moe_state(args, mg_mlp, hf_mlp)
    else:
        _set_mlp_state(mg_mlp, hf_mlp)


def set_layer_state(args, mg_model, hf_model, layer_idx):
    mg_layer = mg_model.decoder.layers[layer_idx]
    hf_layer = hf_model.layers[layer_idx]

    if args.multi_latent_attention:
        set_mla_attn_state(args, mg_layer.self_attention, hf_layer.self_attn)
        hf_layer.input_layernorm.weight.data.copy_(mg_layer.input_layernorm.weight)
    else:
        set_attn_state(args, mg_layer.self_attention, hf_layer.self_attn)
        hf_layer.input_layernorm.weight.data.copy_(mg_layer.self_attention.linear_qkv.layer_norm_weight)

    set_mlp_state(args, mg_layer.mlp, hf_layer.mlp)

    post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
    if 'moe' in mg_layer.mlp.__class__.__name__.lower():
        post_attention_layernorm_weight.data.copy_(mg_layer.pre_mlp_layernorm.weight)
    else:
        post_attention_layernorm_weight.data.copy_(mg_layer.mlp.linear_fc1.layer_norm_weight)


def convert_mcore2hf(hf_model, mg_model):
    args = get_args()
    hf_model.model.embed_tokens.weight.data.copy_(mg_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        lm_head_weight = hf_model.score.weight if args.task_type == 'seq_cls' else hf_model.lm_head.weight
        lm_head_weight.data.copy_(mg_model.output_layer.weight)
    hf_model.model.norm.weight.data.copy_(mg_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state(args, mg_model, hf_model.model, layer_idx)

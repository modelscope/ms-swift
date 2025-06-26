# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.training import get_args


def set_mla_attn_state(args, mg_attn, hf_attn):
    mg_attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)
    if args.q_lora_rank is None:
        mg_attn.linear_q_proj.weight.data.copy_(hf_attn.q_proj.weight)
    else:
        mg_attn.linear_q_down_proj.weight.data.copy_(hf_attn.q_a_proj.weight)
        mg_attn.linear_q_up_proj.weight.data.copy_(hf_attn.q_b_proj.weight)
    mg_attn.linear_kv_down_proj.weight.data.copy_(hf_attn.kv_a_proj_with_mqa.weight)
    mg_attn.linear_kv_up_proj.weight.data.copy_(hf_attn.kv_b_proj.weight)
    if args.qk_layernorm:
        mg_attn.linear_kv_up_proj.layer_norm_weight.data.copy_(hf_attn.kv_a_layernorm.weight)


def set_attn_state(args, mg_attn, hf_attn):
    num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads)

    # Copy weights
    mg_attn.linear_qkv.weight.data.copy_(
        torch.cat([
            hf_attn.q_proj.weight.reshape((num_query_groups, -1, args.hidden_size)),
            hf_attn.k_proj.weight.reshape((num_query_groups, -1, args.hidden_size)),
            hf_attn.v_proj.weight.reshape((num_query_groups, -1, args.hidden_size)),
        ],
                  dim=1).reshape((-1, args.hidden_size)))
    mg_attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)

    # Copy bias
    if args.add_qkv_bias:
        mg_attn.linear_qkv.bias.data.copy_(
            torch.cat([
                hf_attn.q_proj.bias.reshape((num_query_groups, -1)),
                hf_attn.k_proj.bias.reshape((num_query_groups, -1)),
                hf_attn.v_proj.bias.reshape((num_query_groups, -1)),
            ],
                      dim=1).reshape(-1))
    if args.qk_layernorm:
        mg_attn.q_layernorm.weight.data.copy_(hf_attn.q_norm.weight)
        mg_attn.k_layernorm.weight.data.copy_(hf_attn.k_norm.weight)


def _set_mlp_state(mg_mlp, hf_mlp):
    if hasattr(hf_mlp, 'gate_up_proj'):
        mg_mlp.linear_fc1.weight.data.copy_(hf_mlp.gate_up_proj.weight)
    else:
        mg_mlp.linear_fc1.weight.data.copy_(torch.cat([hf_mlp.gate_proj.weight, hf_mlp.up_proj.weight], dim=0))
    mg_mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)


def _set_moe_state(args, mg_mlp, hf_mlp):
    mg_mlp.router.weight.data.copy_(hf_mlp.gate.weight)
    if args.moe_router_enable_expert_bias:
        mg_mlp.router.expert_bias.data.copy_(hf_mlp.gate.e_score_correction_bias)
    if mg_mlp.shared_experts is not None:
        hf_shared_expert = hf_mlp.shared_expert if hasattr(hf_mlp, 'shared_expert') else hf_mlp.shared_experts
        _set_mlp_state(mg_mlp.shared_experts, hf_shared_expert)
        if mg_mlp.shared_experts.gate_weight is not None:
            mg_mlp.shared_experts.gate_weight.data.copy_(hf_mlp.shared_expert_gate.weight)
    for expert_idx in range(args.num_experts):
        _set_mlp_state(mg_mlp.experts.local_experts[expert_idx], hf_mlp.experts[expert_idx])


def set_mlp_state(args, mg_mlp, hf_mlp):
    if 'moe' in mg_mlp.__class__.__name__.lower():
        _set_moe_state(args, mg_mlp, hf_mlp)
    else:
        _set_mlp_state(mg_mlp, hf_mlp)


def set_layer_state(args, mg_model, hf_model, layer_idx):
    mg_layer = mg_model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]
    if args.multi_latent_attention:
        set_mla_attn_state(args, mg_layer.self_attention, hf_layer.self_attn)
        mg_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
    else:
        set_attn_state(args, mg_layer.self_attention, hf_layer.self_attn)
        mg_layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)

    set_mlp_state(args, mg_layer.mlp, hf_layer.mlp)

    post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
    if 'moe' in mg_layer.mlp.__class__.__name__.lower():
        mg_layer.pre_mlp_layernorm.weight.data.copy_(post_attention_layernorm_weight)
    else:
        mg_layer.mlp.linear_fc1.layer_norm_weight.data.copy_(post_attention_layernorm_weight)


def convert_hf2mcore(hf_model, mg_model):
    args = get_args()
    mg_model.embedding.word_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state(args, mg_model, hf_model, layer_idx)

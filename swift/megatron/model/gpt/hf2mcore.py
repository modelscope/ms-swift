# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.training import get_args


def set_attn_state(args, mg_layer, hf_layer):
    mg_attn = mg_layer.self_attention
    hf_attn = hf_layer.self_attn

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


def set_mlp_state(args, mg_layer, hf_layer):
    mg_layer.mlp.linear_fc1.weight.data.copy_(
        torch.cat([hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight], dim=0))
    mg_layer.mlp.linear_fc2.weight.data.copy_(hf_layer.mlp.down_proj.weight)


def set_layer_state(args, mg_model, hf_model, layer_idx):
    mg_layer = mg_model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]
    # self-attention
    set_attn_state(args, mg_layer, hf_layer)
    set_mlp_state(args, mg_layer, hf_layer)
    mg_layer.mlp.linear_fc1.layer_norm_weight.data.copy_(hf_layer.post_attention_layernorm.weight)
    mg_layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)


def convert_hf2mcore(hf_model, mg_model):
    args = get_args()
    mg_model.embedding.word_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state(args, mg_model, hf_model, layer_idx)

# Copyright (c) Alibaba, Inc. and its affiliates.
from megatron.training import get_args


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
        hf_attn.q_norm.weight.data.copy_(mg_attn.q_layernorm.weight)
        hf_attn.k_norm.weight.data.copy_(mg_attn.k_layernorm.weight)


def _set_mlp_state(mg_mlp, hf_mlp):
    ffn_hidden_size = hf_mlp.gate_proj.weight.shape[0]
    hf_mlp.gate_proj.weight.data.copy_(mg_mlp.linear_fc1.weight[:ffn_hidden_size])
    hf_mlp.up_proj.weight.data.copy_(mg_mlp.linear_fc1.weight[ffn_hidden_size:])
    hf_mlp.down_proj.weight.data.copy_(mg_mlp.linear_fc2.weight)


def set_mlp_state(args, mg_mlp, hf_mlp):
    if args.num_experts:
        hf_mlp.gate.weight.data.copy_(mg_mlp.router.weight)
        if mg_mlp.shared_experts is not None:
            hf_mlp.shared_expert_gate.weight.data.copy_(mg_mlp.shared_experts.gate_weight)
        for expert_idx in range(args.num_experts):
            _set_mlp_state(mg_mlp.experts.local_experts[expert_idx], hf_mlp.experts[expert_idx])

        if mg_mlp.shared_experts is not None:
            _set_mlp_state(mg_mlp.shared_experts, hf_mlp.shared_expert)
    else:
        _set_mlp_state(mg_mlp, hf_mlp)


def set_layer_state(args, mg_model, hf_model, layer_idx):
    mg_layer = mg_model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]
    set_attn_state(args, mg_layer.self_attention, hf_layer.self_attn)
    set_mlp_state(args, mg_layer.mlp, hf_layer.mlp)

    post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight
    if args.num_experts:
        post_attention_layernorm_weight.data.copy_(mg_layer.pre_mlp_layernorm.weight)
    else:
        post_attention_layernorm_weight.data.copy_(mg_layer.mlp.linear_fc1.layer_norm_weight)
    hf_layer.input_layernorm.weight.data.copy_(mg_layer.self_attention.linear_qkv.layer_norm_weight)


def convert_mcore2hf(hf_model, mg_model):
    args = get_args()
    hf_model.model.embed_tokens.weight.data.copy_(mg_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        hf_model.lm_head.weight.data.copy_(mg_model.output_layer.weight)
    hf_model.model.norm.weight.data.copy_(mg_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state(args, mg_model, hf_model, layer_idx)

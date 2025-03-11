# Copyright (c) Alibaba, Inc. and its affiliates.

import importlib
from typing import Any, Dict

import torch
from megatron.training import get_args

from swift.llm import Model, ModelGroup, ModelType
from .config import load_config
from .constant import MegatronModelType
from .register import MegatronModelMeta, register_megatron_model


def load_qwen_config(config) -> Dict[str, Any]:
    args_config = load_config(config)
    args_config['swiglu'] = True
    return args_config


def convert_megatron2hf(hf_model, model_provider):
    import toolkits.model_checkpoints_convertor.qwen.hf2mcore_qwen2_dense_and_moe_gqa as module
    from toolkits.model_checkpoints_convertor.qwen.hf2mcore_qwen2_dense_and_moe_gqa import (
        convert_checkpoint_from_megatron_to_transformers, load_megatron_model, check_hf_mg_forward)
    args = get_args()
    module.model_provider = model_provider
    mg_model = load_megatron_model(args)
    convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
    # check_hf_mg_forward(hf_model, mg_model, args)
    return mg_model


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


def convert_hf2megatron(hf_model, mg_model):
    args = get_args()
    mg_model.embedding.word_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state(args, mg_model, hf_model, layer_idx)


def model_provider(pre_process=True, post_process=True):
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt import GPTModel
    args = get_args()
    config = core_transformer_config_from_args(args)
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm,
                                                                        args.qk_layernorm, args.multi_latent_attention)
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling)
    return model


def get_qwen_model_provider():
    return model_provider


register_megatron_model(
    MegatronModelMeta(MegatronModelType.gpt, [ModelType.qwen, ModelType.qwen2, ModelType.qwen2_5], model_provider,
                      load_qwen_config, convert_megatron2hf, convert_hf2megatron))

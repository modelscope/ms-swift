# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from swift.utils import get_logger

logger = get_logger()
config_mapping = {
    'num_layers': ['num_hidden_layers'],
    'hidden_size': ['hidden_size'],
    'ffn_hidden_size': ['intermediate_size'],
    'num_attention_heads': ['num_attention_heads'],
    'num_query_groups': ['num_key_value_heads'],
    'max_position_embeddings': ['max_position_embeddings'],
    'norm_epsilon': ['rms_norm_eps'],
    'rotary_base': ['rope_theta'],
    'padded_vocab_size': ['vocab_size'],
    'attention_dropout': ['attention_dropout'],
    'untie_embeddings_and_output_weights': ['tie_word_embeddings'],
    'swiglu': ['hidden_act'],
    'add_qkv_bias': ['attention_bias', 'qkv_bias', 'use_bias'],
    'disable_bias_linear': ['mlp_bias'],
    'kv_channels': ['head_dim', 'v_head_dim'],
    'architectures': ['architectures'],
    # moe
    'moe_ffn_hidden_size': ['moe_intermediate_size'],
    'moe_shared_expert_intermediate_size': ['shared_expert_intermediate_size'],
    'moe_router_topk': ['num_experts_per_tok', 'n_group', 'moe_topk', 'moe_k'],
    'num_experts': ['num_experts', 'n_routed_experts', 'moe_num_experts'],
    'moe_router_pre_softmax': ['norm_topk_prob'],
    # deepseek
    'q_lora_rank': ['q_lora_rank'],
    'kv_lora_rank': ['kv_lora_rank'],
    'moe_router_score_function': ['scoring_func'],
    'qk_head_dim': ['qk_nope_head_dim'],
    'qk_pos_emb_head_dim': ['qk_rope_head_dim'],
    'moe_router_topk_scaling_factor': ['routed_scaling_factor'],
    'qk_layernorm': ['use_qk_norm'],
    # qwen3_next
    'linear_num_value_heads': ['linear_num_value_heads'],
    'linear_num_key_heads': ['linear_num_key_heads'],
    'linear_key_head_dim': ['linear_key_head_dim'],
    'linear_value_head_dim': ['linear_value_head_dim'],
    'linear_conv_kernel_dim': ['linear_conv_kernel_dim'],
    'full_attention_interval': ['full_attention_interval'],
    # other
    'original_max_position_embeddings': ['original_max_position_embeddings'],
    'partial_rotary_factor': ['partial_rotary_factor'],
    'first_k_dense_replace': ['first_k_dense_replace', 'moe_layer_start_index'],
    'n_shared_experts': ['n_shared_experts', 'num_shared_expert', 'moe_num_shared_experts'],
}


def convert_hf_config(config, _internal_call=False) -> Dict[str, Any]:
    megatron_config = {}
    for k, hf_keys in config_mapping.items():
        for hf_k in hf_keys:
            if hasattr(config, hf_k):
                hf_v = getattr(config, hf_k)
                if hf_v is None:
                    continue
                if k == 'rotary_base':
                    megatron_config[k] = int(hf_v)
                elif k in {'untie_embeddings_and_output_weights', 'disable_bias_linear', 'moe_router_pre_softmax'}:
                    megatron_config[k] = not hf_v
                elif k == 'swiglu':
                    if hf_v == 'silu':
                        megatron_config[k] = True
                else:
                    if k == 'kv_lora_rank':
                        megatron_config['multi_latent_attention'] = True
                    elif k == 'architectures':
                        if _internal_call:
                            k = 'llm_architectures'
                    megatron_config[k] = hf_v
                break
    for key in ['text_config', 'llm_config', 'thinker_config']:
        if hasattr(config, key):
            megatron_config.update(convert_hf_config(getattr(config, key), _internal_call=True))
    # compat llama3
    if getattr(config, 'rope_scaling', None) is not None:
        if isinstance(config.rope_scaling, int):
            megatron_config['rope_scaling'] = {'factor': config.rope_scaling, 'type': 'linear'},
        elif isinstance(config.rope_scaling, dict):
            megatron_config['rope_scaling'] = config.rope_scaling
    return megatron_config

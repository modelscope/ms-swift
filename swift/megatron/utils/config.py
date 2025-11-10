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
    'moe_router_topk': ['num_experts_per_tok', 'moe_topk', 'moe_k'],
    'moe_router_num_groups': ['n_group'],
    'moe_router_group_topk': ['topk_group'],
    'num_experts': ['num_experts', 'n_routed_experts', 'moe_num_experts'],
    'moe_router_pre_softmax': ['norm_topk_prob'],
    # deepseek
    'q_lora_rank': ['q_lora_rank'],
    'kv_lora_rank': ['kv_lora_rank'],
    'moe_router_score_function': ['scoring_func'],
    'moe_router_bias_update_rate': ['aux_loss_alpha'],
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


def _convert_config(config, _internal_call=False) -> Dict[str, Any]:
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
            megatron_config.update(_convert_config(getattr(config, key), _internal_call=True))
    # compat llama3
    if getattr(config, 'rope_scaling', None) is not None:
        if isinstance(config.rope_scaling, int):
            megatron_config['rope_scaling'] = {'factor': config.rope_scaling, 'type': 'linear'},
        elif isinstance(config.rope_scaling, dict):
            megatron_config['rope_scaling'] = config.rope_scaling
    return megatron_config


def convert_hf_config(config) -> Dict[str, Any]:
    res = _convert_config(config)
    architectures = res.get('architectures')
    if isinstance(architectures, list) and architectures:
        architectures = architectures[0]
        res['architectures'] = architectures
    llm_architectures = res.get('llm_architectures') or architectures
    if isinstance(llm_architectures, list) and llm_architectures:
        llm_architectures = llm_architectures[0]
        res['llm_architectures'] = llm_architectures

    first_k_dense_replace = res.pop('first_k_dense_replace', None)
    n_shared_experts = res.pop('n_shared_experts', None)
    if llm_architectures in {'Qwen3ForCausalLM', 'Qwen3MoeForCausalLM', 'Qwen3NextForCausalLM'} or architectures in {
            'Qwen3OmniMoeForConditionalGeneration', 'Qwen3VLForConditionalGeneration',
            'Qwen3VLMoeForConditionalGeneration'
    }:
        res['qk_layernorm'] = True
    if llm_architectures in {'Qwen2MoeForCausalLM', 'Qwen3MoeForCausalLM', 'Qwen3NextForCausalLM'} or architectures in {
            'Qwen3OmniMoeForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration'
    }:
        res.pop('ffn_hidden_size', None)
        if llm_architectures in {'Qwen2MoeForCausalLM', 'Qwen3NextForCausalLM'}:
            res['use_shared_expert_gate'] = True
    if llm_architectures in {
            'DeepseekForCausalLM',
            'DeepseekV2ForCausalLM',
            'DeepseekV3ForCausalLM',
            'Dots1ForCausalLM',
    } or architectures == 'KimiVLForConditionalGeneration':
        if llm_architectures != 'DeepseekForCausalLM':
            res['qk_layernorm'] = True
        res['moe_router_load_balancing_type'] = 'seq_aux_loss'
        res.pop('num_query_groups', None)  # https://github.com/NVIDIA/Megatron-LM/issues/1475
        if llm_architectures == 'Dots1ForCausalLM':
            res['moe_router_score_function'] = 'sigmoid'
    elif llm_architectures == 'HunYuanMoEV1ForCausalLM':
        # Since HunYuan’s attention applies RoPE before using q/k_layernorm,
        # which is incompatible with megatron-core, support is not provided here.
        res['n_shared_experts'] = n_shared_experts
        for key in ['moe_ffn_hidden_size', 'n_shared_experts', 'moe_router_topk']:
            val = res.get(key)
            if isinstance(val, list) and val and min(val) == max(val):
                res[key] = val[0]
        n_shared_experts = res.pop('n_shared_experts')
    elif llm_architectures in {'Ernie4_5_ForCausalLM', 'Ernie4_5_MoeForCausalLM'}:
        res['rotary_interleaved'] = True
    elif llm_architectures == 'Glm4MoeForCausalLM' or architectures == 'Glm4vMoeForConditionalGeneration':
        res['moe_router_score_function'] = 'sigmoid'
    elif llm_architectures == 'Qwen3NextForCausalLM':
        full_attention_interval = res.pop('full_attention_interval')
        num_layers = res['num_layers']
        res['layer_types'] = [
            'full_attention' if (i + 1) % full_attention_interval == 0 else 'linear_attention'
            for i in range(num_layers)
        ]
    if (res.get('rope_scaling') or {}).get('mrope_section') is not None:
        res['position_embedding_type'] = 'mrope'
        res['mrope_section'] = res['rope_scaling']['mrope_section']
        mrope_interleaved = res['rope_scaling'].get('mrope_interleaved', False) or res['rope_scaling'].get(
            'interleaved', False)
        res['mrope_interleaved'] = mrope_interleaved

    if first_k_dense_replace is not None:
        res['moe_layer_freq'] = f'[0]*{first_k_dense_replace}+[1]*{res["num_layers"] - first_k_dense_replace}'
    if res.get('moe_router_score_function', 'softmax') == 'sigmoid':
        res['moe_router_enable_expert_bias'] = True
    if n_shared_experts is not None and 'moe_shared_expert_intermediate_size' not in res:
        res['moe_shared_expert_intermediate_size'] = n_shared_experts * res['moe_ffn_hidden_size']
    return res

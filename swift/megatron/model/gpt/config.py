from typing import Any, Dict

from ..config import convert_hf_config


def convert_gpt_hf_config(config) -> Dict[str, Any]:
    res = convert_hf_config(config)
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
    if llm_architectures in {'Qwen3ForCausalLM', 'Qwen3MoeForCausalLM'}:
        res['qk_layernorm'] = True
    if llm_architectures in {'Qwen2MoeForCausalLM', 'Qwen3MoeForCausalLM'}:
        res.pop('ffn_hidden_size', None)
        if llm_architectures == 'Qwen2MoeForCausalLM':
            res['use_shared_expert_gate'] = True
    if llm_architectures in {
            'DeepseekForCausalLM', 'DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM', 'Dots1ForCausalLM'
    }:
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

    if (res.get('rope_scaling') or {}).get('mrope_section') is not None:
        res['position_embedding_type'] = 'mrope'
        res['mrope_section'] = res['rope_scaling']['mrope_section']

    if first_k_dense_replace is not None:
        res['moe_layer_freq'] = f'[0]*{first_k_dense_replace}+[1]*{res["num_layers"] - first_k_dense_replace}'
    if res.get('moe_router_score_function', 'softmax') == 'sigmoid':
        res['moe_router_enable_expert_bias'] = True
    if n_shared_experts is not None and 'moe_shared_expert_intermediate_size' not in res:
        res['moe_shared_expert_intermediate_size'] = n_shared_experts * res['moe_ffn_hidden_size']
    return res

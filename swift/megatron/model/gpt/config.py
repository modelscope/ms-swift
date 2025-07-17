from typing import Any, Dict

from ..config import convert_hf_config


def convert_gpt_hf_config(config) -> Dict[str, Any]:
    res = convert_hf_config(config)
    architectures = res.get('architectures')
    first_k_dense_replace = res.pop('first_k_dense_replace', None)
    n_shared_experts = res.pop('n_shared_experts', None)
    if isinstance(architectures, list) and architectures:
        architectures = architectures[0]
        res['architectures'] = architectures
        if architectures in {'Qwen3ForCausalLM', 'Qwen3MoeForCausalLM'}:
            res['qk_layernorm'] = True
        if architectures in {'Qwen2MoeForCausalLM', 'Qwen3MoeForCausalLM'}:
            res.pop('ffn_hidden_size', None)
            if architectures == 'Qwen2MoeForCausalLM':
                res['use_shared_expert_gate'] = True
        if architectures in {
                'DeepseekForCausalLM', 'DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM', 'Dots1ForCausalLM'
        }:
            if architectures != 'DeepseekForCausalLM':
                res['qk_layernorm'] = True
            res['moe_router_load_balancing_type'] = 'seq_aux_loss'
            res.pop('num_query_groups', None)  # https://github.com/NVIDIA/Megatron-LM/issues/1475
            if architectures == 'Dots1ForCausalLM':
                res['moe_router_score_function'] = 'sigmoid'
            if res.get('moe_router_score_function', 'softmax') == 'sigmoid':
                res['moe_router_enable_expert_bias'] = True
            res['moe_layer_freq'] = f'[0]*{first_k_dense_replace}+[1]*{res["num_layers"] - first_k_dense_replace}'
    if architectures == 'HunYuanMoEV1ForCausalLM':
        # Since HunYuan’s attention applies RoPE before using q/k_layernorm,
        # which is incompatible with megatron-core, support is not provided here.
        res['n_shared_experts'] = n_shared_experts
        for key in ['moe_ffn_hidden_size', 'n_shared_experts', 'moe_router_topk']:
            val = res.get(key)
            if isinstance(val, list) and val and min(val) == max(val):
                res[key] = val[0]
        n_shared_experts = res.pop('n_shared_experts')
    if architectures in {'Ernie4_5_ForCausalLM', 'Ernie4_5_MoeForCausalLM'}:
        res['rotary_interleaved'] = True
        if architectures == 'Ernie4_5_MoeForCausalLM':
            res['moe_layer_freq'] = f'[0]*{first_k_dense_replace}+[1]*{res["num_layers"] - first_k_dense_replace}'
    if n_shared_experts is not None and 'moe_shared_expert_intermediate_size' not in res:
        res['moe_shared_expert_intermediate_size'] = n_shared_experts * res['moe_ffn_hidden_size']
    return res

from typing import Any, Dict

from ..config import convert_hf_config


def convert_gpt_hf_config(config) -> Dict[str, Any]:
    res = convert_hf_config(config)
    architectures = res.get('architectures')
    if isinstance(architectures, list) and architectures:
        architectures = architectures[0]
        res['architectures'] = architectures
        if architectures in {'Qwen3ForCausalLM', 'Qwen3MoeForCausalLM'}:
            res['qk_layernorm'] = True
        if architectures in {'Qwen2MoeForCausalLM', 'Qwen3MoeForCausalLM'}:
            res.pop('ffn_hidden_size', None)
            if architectures == 'Qwen2MoeForCausalLM':
                res['use_shared_expert_gate'] = True
        if architectures == 'DeepseekV3ForCausalLM':
            res['qk_layernorm'] = True
            res['moe_router_load_balancing_type'] = 'seq_aux_loss'
            res['moe_router_enable_expert_bias'] = True
    return res

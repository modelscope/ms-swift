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
    return res

from typing import Any, Dict

from ..config import convert_hf_config


def convert_gpt_hf_config(config) -> Dict[str, Any]:
    res = convert_hf_config(config)
    model_type = res.get('model_type')
    if model_type in {'qwen3', 'qwen3_moe'}:
        res['qk_layernorm'] = True
    if model_type in {'qwen2_moe', 'qwen3_moe'}:
        res.pop('ffn_hidden_size', None)
    return res

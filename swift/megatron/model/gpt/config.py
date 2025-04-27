from typing import Any, Dict

from ..config import convert_hf_config


def convert_gpt_hf_config(config) -> Dict[str, Any]:
    res = convert_hf_config(config)
    model_type = res.get('model_type')
    if model_type == 'qwen3':
        res['qk_layernorm'] = True
    elif model_type in {'qwen2_moe', 'qwen3_moe'}:
        res.pop('ffn_hidden_size', None)
    return res

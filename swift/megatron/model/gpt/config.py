from ..config import convert_hf_config
from typing import Dict, Any

def convert_gpt_hf_config(config) -> Dict[str, Any]:
    res = convert_hf_config(config)
    if res.get('model_type') == 'qwen3':
        res['qk_layernorm'] = True
    return res

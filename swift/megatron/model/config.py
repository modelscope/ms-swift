from swift.llm import ModelInfo
from typing import Dict, Any

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
    'attention_dropout': ['attention_dropout']
}

def load_config(model_info: ModelInfo) -> Dict[str, Any]:
    model_config = model_info.config
    megatron_config = {}
    for k, value in config_mapping.items():
        for v in value:
            assert hasattr(model_config, v)
            if k == 'rotary_base':
                megatron_config[k] = int(getattr(model_config, v))
            else:
                megatron_config[k] = getattr(model_config, v)
    return megatron_config

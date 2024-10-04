# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Tuple
from transformers import PretrainedConfig

from swift.utils import deep_getattr


class ConfigReader:
    """This class is used to read config from config.json(maybe params.json also)"""

    @staticmethod
    def _get_config_attr(config: PretrainedConfig, attr_name: str) -> Optional[Tuple[PretrainedConfig, Any]]:
        for key in [None, 'language_config', 'llm_config', 'text_config']:
            if key is not None:
                config = getattr(config, key, None)
            value = deep_getattr(config, attr_name)
            if value is not None:
                return config, value

    @staticmethod
    def get_config_attr(config, attr_name: str) -> Optional[Any]:
        value = ConfigReader._get_config_attr(config, attr_name) or (None, None)
        return value[1]

    @staticmethod
    def set_config_attr(config, attr_name: str, value: Any) -> None:
        config, _ = ConfigReader._get_config_attr(config, attr_name) or (config, None)
        setattr(config, attr_name, value)

    @staticmethod
    def set_rope_scaling(config: PretrainedConfig, rope_scaling: Dict[str, Any]):
        """Set rope scaling to the config"""
        # [TODO:check]
        ConfigReader.set_config_attr(config, 'rope_scaling', rope_scaling)

    @staticmethod
    def get_rope_scaling(config: PretrainedConfig) -> Dict[str, Any]:
        """Get rope scaling from the config"""
        # [TODO:check]
        return ConfigReader.get_config_attr(config, 'rope_scaling')

    @staticmethod
    def get_max_model_len(config: PretrainedConfig) -> Optional[int]:
        """Get the max length supported by the model"""
        INF = int(1e9)
        max_model_len = INF

        possible_keys = [
            'seq_length',  # qwen, chatglm
            'max_position_embeddings',  # qwen1.5, llama2
            'n_positions',  # polylm, phi-2
            'model_max_length',  # baichuan2
            # others
            'seq_len',
            'max_seq_len',
            'max_sequence_length',
            'max_seq_length',
        ]
        for key in possible_keys:
            max_len_key = ConfigReader.get_config_attr(config, key)
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        if max_model_len == INF:
            max_model_len = None

        return max_model_len

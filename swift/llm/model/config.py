import os.path
from typing import Dict, Any, Optional

from modelscope import AutoConfig
from transformers import PretrainedConfig

from swift.llm.model.loader import safe_snapshot_download


class ConfigReader:

    @staticmethod
    def read_config(key, model_type, model_id_or_path, revision):
        model_dir = safe_snapshot_download(model_type, model_id_or_path, revision, download_model=False)
        if os.path.exists(os.path.join(model_dir, 'config.json')):
            return ConfigReader.read_config_from_hf(key, model_dir)
        else:
            # For Mistral
            raise NotImplementedError

    @staticmethod
    def read_config_from_hf(key, model_dir):
        config = AutoConfig.from_pretrained(model_dir)
        for k in key.split('.'):
            config = getattr(config, k, None)
            if config is None:
                return None
        return config

    @staticmethod
    def set_rope_scaling(config: PretrainedConfig, rope_scaling: Dict[str, Any]):
        for k in ['language_config', 'llm_config', 'text_config']:
            llm_config = getattr(config, k, None)
            if llm_config is not None:
                config = llm_config
                break

        if getattr(config, 'rope_scaling', None):
            rope_scaling['factor'] = max(config.rope_scaling.get('factor', -1), rope_scaling['factor'])
            rope_scaling = {**config.rope_scaling, **rope_scaling}
        config.rope_scaling = rope_scaling

    @staticmethod
    def get_rope_scaling(config: PretrainedConfig):
        for k in ['language_config', 'llm_config', 'text_config']:
            llm_config = getattr(config, k, None)
            if llm_config is not None:
                config = llm_config
                break

        return getattr(config, 'rope_scaling')

    @staticmethod
    def get_max_model_len(config: PretrainedConfig, ignore_rope_scaling=False) -> Optional[int]:
        INF = int(1e9)
        max_model_len = INF
        for k in ['language_config', 'llm_config', 'text_config']:
            llm_config = getattr(config, k, None)
            if llm_config is not None:
                config = llm_config
                break

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
            max_len_key = getattr(config, key, None)
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        if max_model_len == INF:
            max_model_len = None

        if (not ignore_rope_scaling and max_model_len and getattr(config, 'rope_scaling', None)
                and config.rope_scaling.get('factor')):
            max_model_len = max(int(max_model_len * config.rope_scaling.get('factor')), max_model_len)
        return max_model_len

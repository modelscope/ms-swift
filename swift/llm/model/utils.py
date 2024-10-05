# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import torch.distributed as dist
from datasets.utils.filelock import FileLock
from modelscope.hub.utils.utils import get_cache_dir
from transformers import PretrainedConfig

from swift import get_logger
from swift.hub import HFHub, MSHub
from swift.utils import deep_getattr, is_dist, is_dist_ta, safe_ddp_context

logger = get_logger()


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

    @staticmethod
    def get_quant_method(config: PretrainedConfig) -> Literal['gptq', 'awq', 'bnb', 'aqlm', None]:
        # [TODO:hqq,eetq]
        quantization_config = dict(getattr(config, 'quantization_config'))
        quant_method = quantization_config.get('quant_method')
        if quant_method in {'gptq', 'awq'}:
            return quant_method
        elif quant_method == 'bitsandbytes':
            return 'bnb'
        else:
            return None


def safe_snapshot_download(model_id_or_path: str,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           use_hf: bool = False,
                           ignore_file_pattern: Optional[List[str]] = None,
                           **kwargs) -> str:
    """Download model protected by DDP context

    Args:
        model_id_or_path: The model id or model path
        revision: The model revision
        download_model: Download model bin/safetensors files or not
        use_hf: use huggingface or modelscope

    Returns:
        model_dir
    """
    if (is_dist() or is_dist_ta()) and not dist.is_initialized():
        # Distributed but uninitialized
        lock_dir = os.path.join(get_cache_dir(), 'lockers')
        file_path = hashlib.md5(model_id_or_path.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(lock_dir, file_path)
        context = FileLock(file_path)
    else:
        context = safe_ddp_context()
    hub = HFHub if use_hf else MSHub
    with context:
        if os.path.exists(model_id_or_path):
            model_dir = model_id_or_path
        else:
            if model_id_or_path[:1] in {'~', '/'}:  # startswith
                raise ValueError(f"path: '{model_id_or_path}' not found")
            model_dir = hub.download_model(model_id_or_path, revision, download_model, ignore_file_pattern, **kwargs)

        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


_T = TypeVar('_T')


class AttnImpl:
    flash_attn = 'flash_attn'
    sdpa = 'sdpa'
    eager = 'eager'
    auto = 'auto'

    @staticmethod
    def to_use_flash_attn(attn_impl: Optional[str], default: _T = None) -> Union[bool, _T]:
        if attn_impl in {'auto', None}:
            return default
        return attn_impl == AttnImpl.flash_attn

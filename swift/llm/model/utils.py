# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
from datasets.utils.filelock import FileLock
from modelscope.hub.utils.utils import get_cache_dir
from transformers import PretrainedConfig

from swift import get_logger
from swift.hub import HFHub, MSHub, default_hub
from swift.utils import deep_getattr, is_dist, is_dist_ta, safe_ddp_context

logger = get_logger()


class HfConfigFactory:
    """This class is used to read config from config.json(maybe params.json also)"""

    @staticmethod
    def _get_config_attr(config: PretrainedConfig, attr_name: str) -> Optional[Tuple[PretrainedConfig, Any]]:
        for key in [None, 'language_config', 'llm_config', 'text_config']:
            if key is not None:
                config = getattr(config, key, None)
            value = deep_getattr(config, attr_name, None)
            if value is not None:
                return config, value

    @staticmethod
    def get_config_attr(config, attr_name: str) -> Optional[Any]:
        value = HfConfigFactory._get_config_attr(config, attr_name) or (None, None)
        return value[1]

    @staticmethod
    def get_torch_dtype(config) -> Optional[torch.dtype]:
        for key in ['torch_dtype', 'params_dtype']:
            torch_dtype = HfConfigFactory.get_config_attr(config, key)
            if torch_dtype is None:
                continue
            return HfConfigFactory._to_torch_dtype(torch_dtype)

    @staticmethod
    def set_config_attr(config, attr_name: str, value: Any) -> None:
        config, _ = HfConfigFactory._get_config_attr(config, attr_name) or (config, None)
        setattr(config, attr_name, value)

    @staticmethod
    def set_rope_scaling(config: PretrainedConfig, rope_scaling: Dict[str, Any]):
        """Set rope scaling to the config"""
        # [TODO:check]
        HfConfigFactory.set_config_attr(config, 'rope_scaling', rope_scaling)

    @staticmethod
    def get_rope_scaling(config: PretrainedConfig) -> Dict[str, Any]:
        """Get rope scaling from the config"""
        # [TODO:check]
        return HfConfigFactory.get_config_attr(config, 'rope_scaling')

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
            max_len_key = HfConfigFactory.get_config_attr(config, key)
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        if max_model_len == INF:
            max_model_len = None

        return max_model_len

    @staticmethod
    def compat_zero3(config) -> None:
        value = HfConfigFactory.get_config_attr(config, 'hidden_size')
        config.hidden_size = value

    @staticmethod
    def _to_torch_dtype(torch_dtype: Union[str, torch.dtype]) -> torch.dtype:
        if isinstance(torch_dtype, str):
            torch_dtype = eval(f'torch.{torch_dtype}')
        return torch_dtype

    @staticmethod
    def get_quant_info(config: PretrainedConfig) -> Dict[str, Any]:
        """Get quant_method, quant_bits, dtype. not support hqq/eetq now, support awq/gptq/bnb/aqlm"""
        quantization_config = dict(getattr(config, 'quantization_config'))
        quant_method = quantization_config.get('quant_method')
        res = {}
        if quant_method in {'gptq', 'awq', 'aqlm'}:
            res['quant_method'] = quant_method
            res['torch_dtype'] = torch.float16
            bits = quantization_config.get('bits')
            if bits is not None:
                res['bits'] = bits
        elif quant_method == 'bitsandbytes':
            res['quant_method'] = quant_method
            load_in_4bit = quantization_config.get('load_in_4bit')
            load_in_8bit = quantization_config.get('load_in_8bit')
            bnb_4bit_compute_dtype = quantization_config.get('bnb_4bit_compute_dtype')
            if load_in_4bit:
                res['bits'] = 4
            elif load_in_8bit:
                res['bits'] = 8
            res['torch_dtype'] = HfConfigFactory._to_torch_dtype(bnb_4bit_compute_dtype)
        return res

    @staticmethod
    def get_matched_model_types(config: PretrainedConfig, model_dir: Optional[str] = None) -> List[str]:
        """Get possible model_type."""
        # get possible model_types based on the model architecture.
        from .register import get_arch_mapping
        arch_mapping = get_arch_mapping()
        model_name = None
        if model_dir is not None:
            model_name = model_dir.rsplit('/', 1)[-1].lower()
        arch = config.architectures[0]
        model_type_dict: Dict[str, List[str]] = arch_mapping[arch]
        model_type_list = list(model_type_dict.keys())
        if len(model_type_list) == 1 or model_dir is None:
            return model_type_list
        # Filter again based on model_dir.
        model_type_dict_reversed = {}
        for model_type, model_names in model_type_dict.items():
            model_type_dict_reversed.update({model_name.lower(): model_type for model_name in model_names})
        model_type = model_type_dict_reversed.get(model_name)
        if model_type is None:
            return model_type_list
        return [model_type]


def safe_snapshot_download(model_id_or_path: str,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           use_hf: Optional[bool] = None,
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
    hub = {True: HFHub, False: MSHub, None: default_hub}[use_hf]
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

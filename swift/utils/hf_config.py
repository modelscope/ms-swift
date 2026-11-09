# Copyright (c) ModelScope Contributors. All rights reserved.

import importlib
import torch
from transformers import PretrainedConfig
from typing import Any, Dict, List, Optional, Tuple, Union

from .logger import get_logger
from .utils import deep_getattr

logger = get_logger()


class HfConfigFactory:
    llm_keys = ['language_config', 'llm_config', 'text_config']
    vision_keys = ['vit_config', 'vision_config', 'audio_config']
    """This class is used to read config from config.json(maybe params.json also)"""

    @staticmethod
    def get_torch_dtype(config: Union[PretrainedConfig, Dict[str, Any]],
                        quant_info: Dict[str, Any]) -> Optional[torch.dtype]:
        for key in ['torch_dtype', 'params_dtype']:
            torch_dtype = HfConfigFactory.get_config_attr(config, key)
            if torch_dtype is not None:
                break
        torch_dtype = HfConfigFactory.to_torch_dtype(torch_dtype)
        if torch_dtype is None:
            torch_dtype = quant_info.get('torch_dtype')
        return torch_dtype

    @staticmethod
    def get_text_config(config):
        for key in HfConfigFactory.llm_keys:
            value = getattr(config, key, None)
            if isinstance(value, PretrainedConfig):
                return value
        return config

    @staticmethod
    def _get_config_attrs(config: Union[PretrainedConfig, Dict[str, Any]],
                          attr_name: str,
                          include_vit: bool = False,
                          parent_key: Optional[str] = None) -> List[Tuple[PretrainedConfig, Any]]:
        res = []
        if isinstance(config, dict):
            keys = config.keys()
        elif isinstance(config, PretrainedConfig):
            keys = dir(config)
        else:
            return []
        config_keys = [None] + HfConfigFactory.llm_keys
        if include_vit:
            config_keys += HfConfigFactory.vision_keys
        if attr_name in keys and parent_key in config_keys:
            res.append((config, deep_getattr(config, attr_name)))

        for k in keys:
            if k.endswith('_config') and k != 'talker_config':
                if isinstance(config, dict):
                    v = config[k]
                else:
                    v = getattr(config, k)
                res += HfConfigFactory._get_config_attrs(v, attr_name, include_vit, k)
        return res

    @staticmethod
    def is_moe_model(config) -> bool:
        if 'Moe' in config.__class__.__name__:
            return True
        for key in ['num_experts', 'num_experts_per_tok', 'moe_intermediate_size']:
            if HfConfigFactory.get_config_attr(config, key):
                return True
        return False

    @staticmethod
    def is_multimodal(config) -> bool:
        if isinstance(config, dict):
            keys = config.keys()
        elif isinstance(config, PretrainedConfig):
            keys = dir(config)
        else:
            keys = []
        keys = set(keys)
        for key in (HfConfigFactory.llm_keys + HfConfigFactory.vision_keys + ['thinker_config']):
            if key in keys:
                return True
        return False

    @staticmethod
    def get_native_config_class(model_type: str) -> Optional[type]:
        """Return the Transformers config class, ignoring `AutoConfig.register` overrides."""
        from transformers import PretrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        cls_name = CONFIG_MAPPING_NAMES.get(model_type)
        if not cls_name:
            return None
        import transformers
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            try:
                from transformers.models.auto.configuration_auto import model_type_to_module_name
                module = importlib.import_module(f'transformers.models.{model_type_to_module_name(model_type)}')
            except Exception:
                return None
            cls = getattr(module, cls_name, None)
        if isinstance(cls, type) and issubclass(cls, PretrainedConfig):
            return cls
        return None

    @staticmethod
    def load_hf_config(model_dir: str, *, trust_remote_code: bool = False) -> Optional[PretrainedConfig]:
        """Load `config.json` with the Transformers class instead of an overridden `AutoConfig` one.

        Returns None when the native class cannot be resolved, so callers can fall back to `AutoConfig`.
        """
        try:
            config_dict, _ = PretrainedConfig.get_config_dict(model_dir)
        except Exception:
            return None
        # `auto_map` needs AutoConfig to resolve remote code.
        if config_dict.get('auto_map'):
            return None
        model_type = config_dict.get('model_type')
        if not isinstance(model_type, str):
            return None
        config_cls = HfConfigFactory.get_native_config_class(model_type)
        if config_cls is None:
            return None
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        registered_cls = getattr(CONFIG_MAPPING, '_extra_content', {}).get(model_type)
        if registered_cls is not None and registered_cls is not config_cls:
            logger.info_once(f'{model_type!r} config class is overridden by '
                             f'{registered_cls.__module__}.{registered_cls.__name__}; loading it with '
                             f'{config_cls.__module__}.{config_cls.__name__} instead.')
        try:
            return config_cls.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        except Exception as e:
            logger.debug(f'Failed to load {model_dir!r} with {config_cls.__name__}: {e}')
            return None

    @staticmethod
    def get_config_attr(config: Union[PretrainedConfig, Dict[str, Any]],
                        attr_name: str,
                        include_vit: bool = False) -> Optional[Any]:
        """Get the value of the attribute named attr_name."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name, include_vit)
        if len(attrs) == 0:
            return None
        else:
            return attrs[0][1]

    @staticmethod
    def set_config_attr(config: Union[PretrainedConfig, Dict[str, Any]],
                        attr_name: str,
                        value: Any,
                        include_vit: bool = False,
                        ensure_set: bool = True) -> int:
        """Set all the attr_name attributes to value."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name, include_vit)
        if ensure_set and len(attrs) == 0:
            attrs.append((config, None))
        for config, _ in attrs:
            if isinstance(config, dict):
                config[attr_name] = value
            else:
                setattr(config, attr_name, value)
        return len(attrs)

    @staticmethod
    def del_config_attr(config: Union[PretrainedConfig, Dict[str, Any]],
                        attr_name: str,
                        include_vit: bool = False) -> int:
        """Remove all the attr_name attributes."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name, include_vit)
        for config, _ in attrs:
            if isinstance(config, dict):
                config.pop(attr_name, None)
            elif hasattr(config, attr_name):
                delattr(config, attr_name)
        return len(attrs)

    @staticmethod
    def get_max_model_len(config: Union[PretrainedConfig, Dict[str, Any]]) -> Optional[int]:
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
    def set_max_model_len(config: Union[PretrainedConfig, Dict[str, Any]], value: int):
        """Set the max length supported by the model"""

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
            max_len_value = HfConfigFactory.get_config_attr(config, key)
            if max_len_value is not None:
                HfConfigFactory.set_config_attr(config, key, value)

    @staticmethod
    def compat_zero3(config: PretrainedConfig) -> None:
        value = HfConfigFactory.get_config_attr(config, 'hidden_size')
        try:
            # AttributeError: can't set attribute 'hidden_size'
            config.hidden_size = value
        except AttributeError:
            pass

    @staticmethod
    def to_torch_dtype(torch_dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
        if torch_dtype is None:
            return None
        if isinstance(torch_dtype, str):
            torch_dtype = torch_dtype.replace('torch.', '')
            torch_dtype = getattr(torch, torch_dtype)
        return torch_dtype

    @staticmethod
    def get_quant_info(config: Union[PretrainedConfig, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get quant_method, quant_bits, dtype. not support hqq/eetq now, support awq/gptq/bnb/aqlm"""
        if isinstance(config, dict):
            quantization_config = config.get('quantization_config')
        else:
            quantization_config = getattr(config, 'quantization_config', None)
        if quantization_config is None:
            return
        quantization_config = dict(quantization_config)
        quant_method = quantization_config.get('quant_method')
        res = {}
        if quant_method in {'gptq', 'awq', 'aqlm'}:
            res['quant_method'] = quant_method
            res['torch_dtype'] = torch.float16
            quant_bits = quantization_config.get('bits')
            if quant_bits is not None:
                res['quant_bits'] = quant_bits
        elif quant_method == 'bitsandbytes':
            res['quant_method'] = 'bnb'
            load_in_4bit = quantization_config.get('_load_in_4bit')
            load_in_8bit = quantization_config.get('_load_in_8bit')
            bnb_4bit_compute_dtype = quantization_config.get('bnb_4bit_compute_dtype')
            if load_in_4bit:
                res['quant_bits'] = 4
            elif load_in_8bit:
                res['quant_bits'] = 8
            res['torch_dtype'] = HfConfigFactory.to_torch_dtype(bnb_4bit_compute_dtype)
        elif quant_method == 'hqq':
            res['quant_method'] = quant_method
            res['quant_bits'] = quantization_config['quant_config']['weight_quant_params']['nbits']
        elif quant_method is not None:
            res['quant_method'] = quant_method
        return res or None

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import torch
from accelerate.utils import find_device
from modelscope.hub.utils.utils import get_cache_dir
from torch import nn
from transformers import PretrainedConfig

from swift.hub import get_hub
from swift.llm import to_device
from swift.utils import deep_getattr, get_logger, safe_ddp_context, subprocess_run

logger = get_logger()

_T = TypeVar('_T')


class AttnImpl:
    flash_attn = 'flash_attn'
    sdpa = 'sdpa'
    eager = 'eager'

    attn_impl_keys = ['_attn_implementation', 'attn_implementation', 'llm_attn_implementation']
    use_flash_attn_keys = ['_flash_attn_2_enabled', 'use_flash_attn', '_use_flash_attention_2']

    @staticmethod
    def to_use_flash_attn(attn_impl: Optional[str], auto_value: _T = None) -> Union[bool, _T]:
        if attn_impl is None:
            return auto_value
        return attn_impl == AttnImpl.flash_attn

    @staticmethod
    def update_attn_impl(config: PretrainedConfig,
                         attn_impl: Optional[str],
                         attn_impl_keys: Optional[List[str]] = None) -> None:
        if attn_impl is None:
            return
        logger.info(f'attn_impl: {attn_impl}')
        use_flash_attn = AttnImpl.to_use_flash_attn(attn_impl)
        if use_flash_attn:
            attn_impl = 'flash_attention_2'
        if isinstance(attn_impl_keys, str):
            attn_impl_keys = [attn_impl_keys]
        attn_impl_keys = attn_impl_keys or AttnImpl.attn_impl_keys
        for key in attn_impl_keys:
            HfConfigFactory.set_config_attr(config, key, attn_impl, ensure_set=False)
        for key in AttnImpl.use_flash_attn_keys:
            HfConfigFactory.set_config_attr(config, key, use_flash_attn, ensure_set=False)


@dataclass
class ModelInfo:
    model_type: str
    model_dir: str
    torch_dtype: torch.dtype
    max_model_len: int
    quant_method: Literal['gptq', 'awq', 'bnb', 'aqlm', 'hqq', None]
    quant_bits: int

    # extra
    rope_scaling: Optional[Dict[str, Any]] = None
    config: Optional[PretrainedConfig] = None
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', None] = None
    num_labels: Optional[int] = None

    def __post_init__(self):
        from .register import get_model_name
        self.model_name = get_model_name(self.model_dir)


class HfConfigFactory:
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
    def _get_config_attrs(config: Union[PretrainedConfig, Dict[str, Any]],
                          attr_name: str,
                          parent_key: Optional[str] = None) -> List[Tuple[PretrainedConfig, Any]]:
        res = []
        if isinstance(config, dict):
            keys = config.keys()
        elif isinstance(config, PretrainedConfig):
            keys = dir(config)
        else:
            return []

        value = deep_getattr(config, attr_name, None)
        if value is not None and parent_key in [None, 'language_config', 'llm_config', 'text_config']:
            res.append((config, value))

        for k in keys:
            if k.endswith('_config'):
                if isinstance(config, dict):
                    v = config[k]
                else:
                    v = getattr(config, k)
                res += HfConfigFactory._get_config_attrs(v, attr_name, k)
        return res

    @staticmethod
    def get_config_attr(config: Union[PretrainedConfig, Dict[str, Any]], attr_name: str) -> Optional[Any]:
        """Get the value of the attribute named attr_name."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name)
        if len(attrs) == 0:
            return None
        else:
            return attrs[0][1]

    @staticmethod
    def set_config_attr(config: Union[PretrainedConfig, Dict[str, Any]],
                        attr_name: str,
                        value: Any,
                        ensure_set: bool = True) -> int:
        """Set all the attr_name attributes to value."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name)
        if ensure_set and len(attrs) == 0:
            attrs.append((config, None))
        for config, _ in attrs:
            if isinstance(config, dict):
                config[attr_name] = value
            else:
                setattr(config, attr_name, value)
        return len(attrs)

    @staticmethod
    def set_model_config_attr(model, attr_name: str, value: Any) -> None:
        for module in model.modules():
            if getattr(module, 'config', None) and getattr(module.config, attr_name, value) != value:
                setattr(module.config, attr_name, value)

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
            torch_dtype = eval(f'torch.{torch_dtype}')
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


def safe_snapshot_download(model_id_or_path: str,
                           revision: Optional[str] = None,
                           download_model: bool = True,
                           use_hf: Optional[bool] = None,
                           hub_token: Optional[str] = None,
                           ignore_patterns: Optional[List[str]] = None,
                           check_local: bool = False,
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
    if check_local:
        model_suffix = model_id_or_path.rsplit('/', 1)[-1]
        if os.path.exists(model_suffix):
            model_dir = os.path.abspath(os.path.expanduser(model_suffix))
            logger.info(f'Loading the model using local model_dir: {model_dir}')
            return model_dir
    if ignore_patterns is None:
        ignore_patterns = [
            '*.zip', '*.gguf', '*.pth', '*.pt', 'consolidated*', 'onnx/*', '*.safetensors.md', '*.msgpack', '*.onnx',
            '*.ot', '*.h5'
        ]
    if not download_model:
        ignore_patterns += ['*.bin', '*.safetensors']
    hub = get_hub(use_hf)
    if model_id_or_path.startswith('~'):
        model_id_or_path = os.path.abspath(os.path.expanduser(model_id_or_path))
    with safe_ddp_context(hash_id=model_id_or_path):
        model_path_to_check = '/'.join(model_id_or_path.split(':', 1))
        if os.path.exists(model_id_or_path):
            model_dir = model_id_or_path
            sub_folder = None
        elif os.path.exists(model_path_to_check):
            model_dir = model_path_to_check
            sub_folder = None
        else:
            if model_id_or_path.startswith('/'):  # startswith
                raise ValueError(f"path: '{model_id_or_path}' not found")
            model_id_or_path = model_id_or_path.split(':', 1)  # get sub_folder
            if len(model_id_or_path) == 1:
                model_id_or_path = [model_id_or_path[0], None]
            model_id_or_path, sub_folder = model_id_or_path
            if sub_folder is not None:
                kwargs['allow_patterns'] = [f"{sub_folder.rstrip('/')}/*"]
            model_dir = hub.download_model(model_id_or_path, revision, ignore_patterns, token=hub_token, **kwargs)

        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if sub_folder:
        model_dir = os.path.join(model_dir, sub_folder)
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


def git_clone_github(github_url: str,
                     local_repo_name: Optional[str] = None,
                     branch: Optional[str] = None,
                     commit_hash: Optional[str] = None) -> str:
    if github_url.endswith('.git'):
        github_url = github_url[:-4]
    git_cache_dir = os.path.join(get_cache_dir(), '_github')
    os.makedirs(git_cache_dir, exist_ok=True)
    if local_repo_name is None:
        github_url = github_url.rstrip('/')
        local_repo_name = github_url.rsplit('/', 1)[1]
    local_repo_path = os.path.join(git_cache_dir, local_repo_name)
    with safe_ddp_context(hash_id=local_repo_path):
        if not os.path.exists(local_repo_path):
            github_url = f'{github_url}.git'
            command = ['git', '-C', git_cache_dir, 'clone', github_url, local_repo_name]
            command_str = f"git -C '{git_cache_dir}' clone '{github_url}' {local_repo_name}"
            if branch is not None:
                command += ['--branch', branch]
                command_str += f' --branch {branch}'
            logger.info(f'Run the command: `{command_str}`')
            subprocess_run(command)

            if commit_hash is not None:
                git_cache_path = os.path.join(git_cache_dir, local_repo_name)
                command = ['git', '-C', git_cache_path, 'reset', '--hard', commit_hash]
                command_str = f"git -C '{git_cache_path}' reset '--hard' {commit_hash}"
                logger.info(f'Run the command: `{command_str}`')
                subprocess_run(command)

        logger.info(f'local_repo_path: {local_repo_path}')
    return local_repo_path


def get_llm_model(model: torch.nn.Module, model_meta=None):
    from swift import SwiftModel
    from peft import PeftModel
    from swift.llm import get_model_arch
    from accelerate.utils import extract_model_from_parallel
    model = extract_model_from_parallel(model)

    if isinstance(model, (SwiftModel, PeftModel)):
        model = model.model
    if model_meta is None:
        model_meta = model.model_meta

    llm_prefix = getattr(get_model_arch(model_meta.model_arch), 'language_model', None)
    if llm_prefix:
        llm_model = deep_getattr(model, llm_prefix[0])
    else:
        llm_model = model

    if 'CausalLM' not in llm_model.__class__.__name__:
        llm_model = model
    return llm_model


def use_submodel_func(model, submodel_name: str, func_list: Optional[List[str]] = None) -> None:
    if func_list is None:
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
    submodel = getattr(model, submodel_name)

    def _get_new_func(func_name: str):
        _old_func = getattr(submodel, func_name).__func__

        @wraps(_old_func)
        def _new_func(self, *args, **kwargs):
            res = _old_func(submodel, *args, **kwargs)
            if func_name == 'forward':
                device = find_device(args)
                if device is None:
                    device = find_device(kwargs)
                if hasattr(res, 'logits'):
                    res.logits = to_device(res.logits, device)
                if hasattr(res, 'loss'):
                    res.loss = to_device(res.loss, device)
                if isinstance(res, dict) and 'last_hidden_state' in res:
                    res['last_hidden_state'] = to_device(res['last_hidden_state'], device)
            return res

        return _new_func

    for key in func_list:
        setattr(model, key, MethodType(_get_new_func(key), model))
        if key == 'generate' and model.device != submodel.device:
            submodel.__class__.device = model.device
        if key == 'forward' and 'generate' in func_list:
            setattr(submodel, key, MethodType(_get_new_func(key), submodel))  # fix device_map


class InitModelStrategy:

    @staticmethod
    def is_uninitialized(param: torch.Tensor) -> bool:
        """
        Check if a parameter is uninitialized or has numerically unstable values.
        Criteria:
            - Tensor has NaN or Inf values
            - Tensor stats (mean or std) are outside reasonable range
        """
        if param.numel() == 0:
            return False

        with torch.no_grad():
            mean_abs = param.abs().mean()
            std = param.std()

            # NaN or Inf
            if not torch.isfinite(mean_abs) or not torch.isfinite(std):
                return True

            # Use empirically safe threshold
            MAX_THRESHOLD = 1e7
            if mean_abs > MAX_THRESHOLD or std > MAX_THRESHOLD:
                return True

            return False

    @staticmethod
    def constant_init(param: torch.Tensor, c: float = 0) -> None:
        nn.init.constant_(param, c)

    @staticmethod
    def uniform_init(param: torch.Tensor, a: float = -0.1, b: float = 0.1) -> None:
        nn.init.uniform_(param, a, b)

    @staticmethod
    def normal_init(param: torch.Tensor, mean: float = 0.0, std: float = 0.01) -> None:
        nn.init.normal_(param, mean, std)

    @staticmethod
    def _init_high_dim(param: torch.Tensor, init_func, *args, **kwargs) -> None:
        """Helper for high-dimensional initialization methods."""
        if param.dim() > 1:
            init_func(param, *args, **kwargs)
        elif param.dim() == 1 and param.size(0) > 0:
            InitModelStrategy.constant_init(param)

    @staticmethod
    def xavier_uniform_init(param: torch.Tensor) -> None:
        InitModelStrategy._init_high_dim(param, nn.init.xavier_uniform_)

    @staticmethod
    def xavier_normal_init(param: torch.Tensor) -> None:
        InitModelStrategy._init_high_dim(param, nn.init.xavier_normal_)

    @staticmethod
    def kaiming_uniform_init(param: torch.Tensor) -> None:
        InitModelStrategy._init_high_dim(
            param, nn.init.kaiming_uniform_, mode='fan_out', nonlinearity='leaky_relu', a=0.1)

    @staticmethod
    def kaiming_normal_init(param: torch.Tensor) -> None:
        InitModelStrategy._init_high_dim(param, nn.init.kaiming_normal_, mode='fan_in', nonlinearity='relu')

    @staticmethod
    def orthogonal_init(param: torch.Tensor) -> None:
        nn.init.orthogonal_(param, gain=1.0)

    _INIT_STRATEGY_MAP = {
        'zero': constant_init,
        'uniform': uniform_init,
        'normal': normal_init,
        'xavier_uniform': xavier_uniform_init,
        'xavier_normal': xavier_normal_init,
        'kaiming_uniform': kaiming_uniform_init,
        'kaiming_normal': kaiming_normal_init,
        'orthogona': orthogonal_init,
    }

    @staticmethod
    def init_parameters(model: nn.Module, init_strategy: str) -> None:
        """Initialize model parameters using the specified strategy.
        Args:
            model: The model whose parameters to initialize
            init_strategy: Name of initialization strategy
        """
        if init_strategy not in InitModelStrategy._INIT_STRATEGY_MAP:
            raise ValueError(f'Unknown initialization strategy: {init_strategy}')

        logger.info(f'initialization strategy: {init_strategy}')

        init_func = InitModelStrategy._INIT_STRATEGY_MAP[init_strategy]

        for name, param in model.named_parameters():
            if InitModelStrategy.is_uninitialized(param):
                logger.info(f'Initializing parameters: {name}.')
                init_func(param)

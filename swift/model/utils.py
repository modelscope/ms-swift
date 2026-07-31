# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import shutil
import torch
import torch.nn.functional as F
from accelerate.utils import find_device
from collections import OrderedDict
from functools import wraps
from packaging import version
from peft import PeftModel
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import (is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_mps_available,
                                is_torch_npu_available, strtobool)
from types import MethodType
from typing import Dict, List, Optional, TypeVar, Union

from swift.utils import (HfConfigFactory, Processor, deep_getattr, get_dist_setting, get_env_args, get_logger, is_mp,
                         to_device)

logger = get_logger()

_T = TypeVar('_T')


class AttnImpl:
    attn_impl_keys = ['_attn_implementation', 'attn_implementation', 'llm_attn_implementation']
    use_flash_attn_keys = ['_flash_attn_2_enabled', 'use_flash_attn', '_use_flash_attention_2']

    @staticmethod
    def to_use_flash_attn(attn_impl: Optional[str], auto_value: _T = None) -> Union[bool, _T]:
        if attn_impl is None:
            return auto_value
        return attn_impl in {'flash_attn', 'flash_attention_2'}

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
            HfConfigFactory.set_config_attr(config, key, attn_impl, include_vit=True, ensure_set=False)
        for key in AttnImpl.use_flash_attn_keys:
            HfConfigFactory.set_config_attr(config, key, use_flash_attn, include_vit=True, ensure_set=False)


def get_llm_model(model: torch.nn.Module, model_meta=None, inner_backbone=True):
    """Get LLM model, this function can be used to get the llm module from a multi-modal model.

    Args:
        model: The model instance
        model_meta: The model_meta information
        inner_backbone: Get inner backbone model, like `QwenModel` or `LlamaModel`

    Returns:

    """
    from accelerate.utils import extract_model_from_parallel

    from swift.tuners import SwiftModel
    model = extract_model_from_parallel(model)

    if isinstance(model, (SwiftModel, PeftModel)):
        model = model.model
    if model_meta is None:
        model_meta = model.model_meta

    llm_prefix = getattr(model_meta.model_arch, 'language_model', None)
    if llm_prefix:
        llm_model = deep_getattr(model, llm_prefix[0])
    else:
        llm_model = model

    if inner_backbone:
        if hasattr(llm_model, 'thinker'):
            llm_model = llm_model.thinker.model
        elif hasattr(llm_model, 'model'):
            llm_model = llm_model.model
    return llm_model


def use_submodel_func(model, submodel_name: str, func_list: Optional[List[str]] = None) -> None:
    if func_list is None:
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
    submodel = getattr(model, submodel_name)

    def _get_new_func(func_name: str):
        # Please ensure the patch to submodel.forward is applied before this function.
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


def get_default_device_map():
    if is_deepspeed_zero3_enabled() or os.environ.get('ACCELERATE_USE_FSDP', 'False') == 'true':
        return None
    local_rank = get_dist_setting()[1]
    if local_rank == -1:
        local_rank = 0
    if is_torch_npu_available():
        return 'auto' if is_mp() else f'npu:{local_rank}'
    elif is_torch_mps_available():
        return f'mps:{local_rank}'
    elif is_torch_cuda_available():
        return 'auto' if is_mp() else f'cuda:{local_rank}'
    else:
        return 'cpu'


def get_default_torch_dtype(torch_dtype: Optional[torch.dtype]):
    # torch_dtype: torch_dtype in config.json
    if torch_dtype is not None:
        return torch_dtype

    try:
        is_bf16_available = is_torch_bf16_gpu_available() or (is_torch_npu_available()
                                                              and torch.npu.is_bf16_supported())
    except Exception:  # noqa
        is_bf16_available = False

    if is_torch_cuda_available() or is_torch_npu_available():
        if is_bf16_available:
            return torch.bfloat16
        else:
            return torch.float16
    else:
        # cpu
        return torch.float32


def _patch_conv3d():
    if hasattr(nn.Conv3d, '_original_forward'):
        return
    nn.Conv3d._original_forward = nn.Conv3d.forward

    def forward(self, x):
        if any(s != k for s, k in zip(self.stride, self.kernel_size)) or any(p != 0 for p in self.padding) or any(
                d != 1 for d in self.dilation) or self.groups != 1:
            raise NotImplementedError(
                'Patched Conv3d only supports stride=kernel_size, padding=0, dilation=1, groups=1')
        N = x.shape[0]
        K = self.kernel_size
        x = x.unfold(2, K[0], K[0]).unfold(3, K[1], K[1]).unfold(4, K[2], K[2])
        D_out, H_out, W_out = x.shape[2:5]
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, self.in_channels * K[0] * K[1] * K[2])
        x = F.linear(x, self.weight.view(self.out_channels, -1), self.bias)
        x = x.view(N, D_out, H_out, W_out, self.out_channels).permute(0, 4, 1, 2, 3)
        return x

    nn.Conv3d.forward = forward
    logger.info('Conv3d patched successfully')


requires_patch = version.parse('2.9.0') <= version.parse(torch.__version__) < version.parse('2.10.0')
if requires_patch:
    _patch_conv3d()


def _get_language_model_prefixes(model: PreTrainedModel) -> List[str]:
    model_arch = getattr(getattr(model, 'model_meta', None), 'model_arch', None)
    prefixes = getattr(model_arch, 'language_model', None) or []
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return prefixes


def _get_language_model_target_prefix(source_prefix: str) -> str:
    if source_prefix == 'language_model':
        return 'model'
    suffix = '.language_model'
    if source_prefix.endswith(suffix):
        return source_prefix[:-len(suffix)]
    raise ValueError(
        f'Cannot export language-model-only checkpoint for language_model prefix `{source_prefix}`. '
        'Only `language_model` and `*.language_model` prefixes are currently supported.')


def get_language_model_state_dict(model: PreTrainedModel) -> Dict[str, torch.Tensor]:
    prefixes = _get_language_model_prefixes(model)
    if not prefixes:
        raise ValueError('`export_language_model_only` requires a multimodal model with language_model prefixes.')

    source_prefix = prefixes[0].rstrip('.')
    target_prefix = _get_language_model_target_prefix(source_prefix)
    state_dict = model.state_dict()
    output_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = None
        if key.startswith(f'{source_prefix}.'):
            new_key = f'{target_prefix}.{key[len(source_prefix) + 1:]}'
        elif key == source_prefix:
            new_key = target_prefix
        else:
            for prefix in prefixes[1:]:
                prefix = prefix.rstrip('.')
                if key == prefix or key.startswith(f'{prefix}.'):
                    new_key = key
                    break
        if new_key is None:
            continue
        if new_key in output_state_dict:
            raise ValueError(f'Duplicate key `{new_key}` while exporting language-model-only checkpoint.')
        output_state_dict[new_key] = value

    if not output_state_dict:
        raise ValueError(f'No language model weights found with prefixes: {prefixes}.')
    return output_state_dict


def _infer_language_model_architectures(model: PreTrainedModel) -> Optional[List[str]]:
    candidates = []
    config = getattr(model, 'config', None)
    if config is not None:
        architectures = getattr(config, 'architectures', None) or []
        candidates.extend(architectures)
    candidates.append(model.__class__.__name__)

    for arch in candidates:
        if arch.endswith('ForCausalLM'):
            return [arch]
        if arch.endswith('ForConditionalGeneration'):
            return [arch[:-len('ForConditionalGeneration')] + 'ForCausalLM']
    return None


def get_language_model_config(model: PreTrainedModel) -> PretrainedConfig:
    text_config = HfConfigFactory.get_text_config(model.config)
    if text_config is model.config:
        raise ValueError('`export_language_model_only` requires a multimodal config with a text config.')

    text_config = copy.deepcopy(text_config)
    architectures = _infer_language_model_architectures(model)
    if architectures is not None:
        text_config.architectures = architectures
    return text_config


def save_language_model_checkpoint(model: PreTrainedModel,
                                   output_dir: str,
                                   *,
                                   safe_serialization: bool = True,
                                   max_shard_size: Union[int, str] = '5GB') -> None:
    try:
        from huggingface_hub import save_torch_state_dict
    except ImportError as e:
        raise ImportError('`export_language_model_only` requires `huggingface_hub.save_torch_state_dict`.') from e

    os.makedirs(output_dir, exist_ok=True)
    state_dict = get_language_model_state_dict(model)
    text_config = get_language_model_config(model)
    text_config.save_pretrained(output_dir)
    generation_config = getattr(model, 'generation_config', None)
    if generation_config is not None:
        generation_config.save_pretrained(output_dir)
    save_torch_state_dict(
        state_dict,
        output_dir,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
        metadata={'format': 'pt'} if safe_serialization else None)


def save_processor_checkpoint(processor: Processor, output_dir: str, *, language_model_only: bool = False) -> None:
    if language_model_only:
        tokenizer = getattr(processor, 'tokenizer', None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            return
    processor.save_pretrained(output_dir)


def save_checkpoint(model: Optional[PreTrainedModel],
                    processor: Processor,
                    output_dir: str,
                    *,
                    safe_serialization: bool = True,
                    max_shard_size: Union[int, str] = '5GB',
                    model_dirs: List[str] = None,
                    additional_saved_files: Optional[List[str]] = None,
                    language_model_only: bool = False) -> None:
    if model is not None:
        if language_model_only:
            save_language_model_checkpoint(
                model, output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
        elif model.__class__.__name__ != 'SentenceTransformer':
            model.save_pretrained(output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
        else:
            model.save_pretrained(output_dir, safe_serialization=safe_serialization)
            # copy sentencetransformers files
            from swift.utils import copy_files_by_pattern
            copy_files_by_pattern(model.model_dir, output_dir, '*.py')
            copy_files_by_pattern(model.model_dir, output_dir, '*.json')
    save_processor_checkpoint(processor, output_dir, language_model_only=language_model_only)

    if model_dirs is None:
        model_dirs = []
    else:
        model_dirs = model_dirs.copy()
    if model and model.model_dir and model.model_dir not in model_dirs:
        model_dirs.append(model.model_dir)
    src_files = (additional_saved_files or []) + ['args.json']
    if not language_model_only:
        src_files.append('preprocessor_config.json')
    for src_file in src_files:
        tgt_path = os.path.join(output_dir, src_file)
        if os.path.exists(tgt_path) and src_file == 'args.json':
            continue
        for model_dir in model_dirs:
            src_path: str = os.path.join(model_dir, src_file)
            if os.path.isfile(src_path):
                shutil.copy(src_path, tgt_path)
                break
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, tgt_path)
                break


def get_ckpt_dir(model_dir: str, adapters_dir: Optional[List[str]]) -> str:
    model_dirs = (adapters_dir or []).copy()
    if model_dir:
        model_dirs.append(model_dir)
    # The adapter takes higher priority.
    ckpt_dir = None
    for model_dir in model_dirs:
        if os.path.exists(os.path.join(model_dir, 'args.json')):
            ckpt_dir = model_dir
            break
    return ckpt_dir

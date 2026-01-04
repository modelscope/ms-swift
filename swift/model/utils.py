# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from accelerate.utils import find_device
from modelscope.hub.utils.utils import get_cache_dir
from torch import nn
from transformers import PretrainedConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import (is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_mps_available,
                                is_torch_npu_available, strtobool)

from swift.hub import get_hub
from swift.llm import to_device
from swift.utils import (deep_getattr, get_dist_setting, get_logger, is_local_master, is_mp, safe_ddp_context,
                         subprocess_run)

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
        with safe_ddp_context(hash_id=model_id_or_path):
            model_dir = hub.download_model(model_id_or_path, revision, ignore_patterns, token=hub_token, **kwargs)

        logger.info(f'Loading the model using model_dir: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if sub_folder:
        model_dir = os.path.join(model_dir, sub_folder)
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


def git_clone_github(github_url: str,
                     *,
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
    github_url = f'{github_url}.git'
    local_repo_path = os.path.join(git_cache_dir, local_repo_name)
    with safe_ddp_context('git_clone', use_barrier=True):
        repo_existed = os.path.exists(local_repo_path)
        if not is_local_master() and repo_existed:
            return local_repo_path
        if repo_existed:
            command = ['git', '-C', local_repo_path, 'fetch']
            subprocess_run(command)
            if branch is not None:
                command = ['git', '-C', local_repo_path, 'checkout', branch]
                subprocess_run(command)
        else:
            command = ['git', '-C', git_cache_dir, 'clone', github_url, local_repo_name]
            if branch is not None:
                command += ['--branch', branch]
            subprocess_run(command)

        if commit_hash is not None:
            command = ['git', '-C', local_repo_path, 'reset', '--hard', commit_hash]
            subprocess_run(command)
        elif repo_existed:
            command = ['git', '-C', local_repo_path, 'pull']
            subprocess_run(command)
    logger.info(f'local_repo_path: {local_repo_path}')
    return local_repo_path


def get_llm_model(model: torch.nn.Module, model_meta=None, inner_backbone=True):
    """Get LLM model, this function can be used to get the llm module from a multi-modal model.

    Args:
        model: The model instance
        model_meta: The model_meta information
        inner_backbone: Get inner backbone model, like `QwenModel` or `LlamaModel`

    Returns:

    """
    from swift.tuners import SwiftModel
    from peft import PeftModel
    from accelerate.utils import extract_model_from_parallel
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
    except:  # noqa
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

    if not hasattr(nn.Conv3d, '_original_forward'):
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


if strtobool(os.getenv('SWIFT_PATCH_CONV3D', 'false')):
    _patch_conv3d()

# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
import shutil
import tempfile
from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from modelscope.hub.utils.utils import get_cache_dir
from peft import PeftModel
from transformers import FeatureExtractionMixin, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from swift.utils import deep_getattr, get_logger
from .utils import Processor

logger = get_logger()


def set_generation_config(model: nn.Module, generation_config: GenerationConfig) -> None:
    old_generation_config = getattr(model, 'generation_config', None)
    old_generation_priority_config = ['no_repeat_ngram_size', 'num_beams']
    if old_generation_config is not None:
        for k, old_v in dir(old_generation_config).items():
            if k.startswith('_'):
                continue
            v = getattr(generation_config, k, None)
            if k in old_generation_priority_config or old_v is not None and v is None:
                setattr(generation_config, k, old_v)
    model.generation_config = generation_config


def find_module_list(model) -> Optional[nn.ModuleList]:
    module_lists = []
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing') or m.__class__.__name__ == 'CheckpointWrapper':
            return
        if (isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) >= 10
                and 'mlp' not in m[0].__class__.__name__.lower()):  # fix moe
            module_lists.append(m)
    if module_lists:
        return max(module_lists, key=lambda x: len(x))


def _kwargs_to_args(func, args, kwargs) -> Optional[List[Any]]:
    parameters = inspect.signature(func).parameters
    args = list(args)
    parameters = list(parameters.items())[len(args):]
    for key, param in parameters:
        if key in kwargs:
            args.append(kwargs[key])
        elif param.default != param.empty:
            args.append(param.default)
        else:
            return
    return args


def _add_gradient_checkpointing(module_list):

    requires_grad = None

    def _new_forward(self, *args, **kwargs):
        nonlocal requires_grad
        if requires_grad is None:
            requires_grad = any(p.requires_grad for p in self.parameters())

        new_args = _kwargs_to_args(self.__old_forward, args, kwargs)
        if new_args is not None and self.gradient_checkpointing and self.training:
            if new_args and isinstance(new_args[0], torch.Tensor) and requires_grad and not new_args[0].requires_grad:
                new_args[0].requires_grad_(True)
            layer_ret = self._gradient_checkpointing_func(self.__old_forward, *new_args)
            logger.info_once('Successfully using dynamic gradient checkpointing.')
        else:
            layer_ret = self.__old_forward(*args, **kwargs)
        return layer_ret

    for module in module_list:
        module.gradient_checkpointing = False
        if hasattr(module, '_old_forward'):  # device_map
            __old_forward = module._old_forward
            module._old_forward = MethodType(_new_forward, module)
        else:
            __old_forward = module.forward
            module.forward = MethodType(_new_forward, module)
        module.__old_forward = __old_forward


def dynamic_gradient_checkpointing(model, including_vit: bool = False) -> None:
    from .model import ModelMeta
    if isinstance(model, PeftModel):
        model = model.model
    model_meta: ModelMeta = getattr(model, 'model_meta', None)
    if model_meta is not None and model_meta.is_multimodal and model_meta.model_arch:
        tower_names = model_meta.model_arch.language_model.copy()
        if including_vit:
            tower_names += model_meta.model_arch.vision_tower
    else:
        tower_names = [None]

    model.supports_gradient_checkpointing = True
    for tower_name in tower_names:
        if tower_name is None:
            model_tower = model
        else:
            model_tower = deep_getattr(model, tower_name)
        model_tower.supports_gradient_checkpointing = True
        module_list = find_module_list(model_tower)
        if module_list is None:
            continue
        _add_gradient_checkpointing(module_list)
        logger.info(f'Automatically add gradient_checkpointing to {model_tower.__class__}.')


TEMP_DIR_POOL = {}


def get_temporary_cache_files_directory(prefix=None):
    if prefix is None:
        import datasets.config
        prefix = datasets.config.TEMP_CACHE_DIR_PREFIX
    global TEMP_DIR_POOL
    if prefix in TEMP_DIR_POOL:
        TEMP_DIR = TEMP_DIR_POOL[prefix]
    else:
        tmp_dir = os.path.join(get_cache_dir(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        kwargs = {}
        parameters = inspect.signature(tempfile.TemporaryDirectory.__init__).parameters
        if 'ignore_cleanup_errors' in parameters:
            kwargs['ignore_cleanup_errors'] = True
        TEMP_DIR = tempfile.TemporaryDirectory(prefix=prefix, dir=tmp_dir, **kwargs)
        logger.info(f'create tmp_dir: {TEMP_DIR.name}')
        TEMP_DIR_POOL[prefix] = TEMP_DIR

    return TEMP_DIR.name


def update_generation_config_eos_token(generation_config, template):
    if generation_config is None:
        return
    stop_words = template.template_meta.stop_words
    eos_token_id = generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = []
    elif isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    modified = False
    for stop_word in stop_words:
        if stop_word is None:
            continue
        if isinstance(stop_word, str):
            stop_word = template._tokenize(stop_word)
        if isinstance(stop_word, (list, tuple)) and len(stop_word) == 1 and stop_word[0] not in eos_token_id:
            eos_token_id.append(stop_word[0])
            modified = True
    if modified:
        generation_config.eos_token_id = eos_token_id

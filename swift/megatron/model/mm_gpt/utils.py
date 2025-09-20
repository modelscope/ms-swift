# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

import torch
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from megatron.training import get_args, get_tokenizer
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ContextManagers

from swift.llm import deep_getattr, get_model_tokenizer
from swift.utils import disable_safe_ddp_context_use_barrier
from ..gpt.config import convert_gpt_hf_config
from ..mm_gpt_model import MultimodalGPTModel
from ..register import MegatronModelMeta


@contextmanager
def patch_hf_initialize_weight():

    _origin_initialize_weight = PreTrainedModel._initialize_weights

    def _initialize_weight(self, *args, **kwargs):
        return

    PreTrainedModel._initialize_weights = _initialize_weight
    try:
        yield
    finally:
        PreTrainedModel._initialize_weights = _origin_initialize_weight


@contextmanager
def patch_device_map_meta(model_cls):
    __origin_init__ = model_cls.__init__

    def __init__(self, *args, **kwargs):
        with torch.device('meta'):
            __origin_init__(self, *args, **kwargs)

    model_cls.__init__ = __init__

    try:
        yield
    finally:
        model_cls.__init__ = __origin_init__


@dataclass
class MMGPTMegatronModelMeta(MegatronModelMeta):
    model_cls: Type[nn.Module] = MultimodalGPTModel
    convert_hf_config: Callable[[PretrainedConfig], Dict[str, Any]] = convert_gpt_hf_config


class HuggingFaceModule(_HuggingFaceModule, ABC):
    module_mapping = {}  # hf -> mcore

    def __init__(self, config, ignore_init_model_cls=None):
        super().__init__(config)
        args = get_args()
        model_dir = args.model_info.model_dir
        attn_impl = getattr(args, 'attn_impl', None) or 'flash_attn'
        kwargs = {'attn_impl': attn_impl} if args.attention_backend.name == 'flash' else {}
        ignore_init_model_cls = ignore_init_model_cls or []
        if not isinstance(ignore_init_model_cls, list):
            ignore_init_model_cls = [ignore_init_model_cls]
        context_list = [patch_device_map_meta(model_cls) for model_cls in ignore_init_model_cls]
        context_list.append(patch_hf_initialize_weight())
        kwargs['model_type'] = args.model_info.model_type
        with ContextManagers(context_list), disable_safe_ddp_context_use_barrier():
            model, _ = get_model_tokenizer(model_dir, args.torch_dtype, return_dummy_model=True, **kwargs)
        self.model_config = model.config
        self.processor = get_tokenizer()
        for hf_prefix, mg_prefix in self.module_mapping.items():
            setattr(self, mg_prefix, deep_getattr(model, hf_prefix))
        self._hf_model = [model]
        self.prepare_model(model)
        self.to('cuda')

    def prepare_model(self, hf_model):
        pass

    @abstractmethod
    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pass

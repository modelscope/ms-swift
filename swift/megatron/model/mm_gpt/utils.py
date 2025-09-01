from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

import torch
from torch import nn
from transformers import PretrainedConfig

from ..gpt.config import convert_gpt_hf_config
from ..mm_gpt_model import MultimodalGPTModel
from ..model_provider import model_provider as model_provider_func
from ..register import MegatronModelMeta, register_megatron_model


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
    model_provider: Callable[[], nn.Module] = model_provider_func
    convert_hf_config: Callable[[PretrainedConfig], Dict[str, Any]] = convert_gpt_hf_config

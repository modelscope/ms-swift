# Copyright (c) Alibaba, Inc. and its affiliates.
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import torch.nn as nn
from transformers import PretrainedConfig

from swift.llm import MODEL_MAPPING
from .model_provider import model_provider as model_provider_func

MEGATRON_MODEL_MAPPING = {}


@dataclass
class MegatronModelMeta:
    megatron_model_type: str
    model_types: List[str]

    convert_mcore2hf: Callable[[nn.Module, nn.Module], None]
    convert_hf2mcore: Callable[[nn.Module, nn.Module], None]

    model_cls: Type[nn.Module]
    convert_hf_config: Callable[[PretrainedConfig], Dict[str, Any]]
    get_transformer_layer_spec: Optional[Callable] = None
    model_provider: Callable[[], nn.Module] = model_provider_func
    visual_cls: Optional[Type[nn.Module]] = None

    extra_args_provider: Optional[Callable[[ArgumentParser], ArgumentParser]] = None


def register_megatron_model(megatron_model_meta: MegatronModelMeta, *, exist_ok: bool = False):
    megatron_model_type = megatron_model_meta.megatron_model_type
    for model_type in megatron_model_meta.model_types:
        model_meta = MODEL_MAPPING[model_type]
        model_meta.support_megatron = True
    if not exist_ok and megatron_model_type in MEGATRON_MODEL_MAPPING:
        raise ValueError(f'The `{megatron_model_type}` has already been registered in the MODEL_MAPPING.')

    MEGATRON_MODEL_MAPPING[megatron_model_type] = megatron_model_meta


_MODEL_META_MAPPING = None


def get_megatron_model_meta(model_type: str) -> Optional[MegatronModelMeta]:
    global _MODEL_META_MAPPING
    if _MODEL_META_MAPPING is None:
        _MODEL_META_MAPPING = {}
        for k, megatron_model_meta in MEGATRON_MODEL_MAPPING.items():
            for _model_type in megatron_model_meta.model_types:
                _MODEL_META_MAPPING[_model_type] = k
    if model_type not in _MODEL_META_MAPPING:
        return
    return MEGATRON_MODEL_MAPPING[_MODEL_META_MAPPING[model_type]]

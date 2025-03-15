# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from transformers import PretrainedConfig

from swift.llm import MODEL_MAPPING, ModelGroup
from swift.llm.model.register import _get_matched_model_meta

MEGATRON_MODEL_MAPPING = {}


@dataclass
class MegatronModelMeta:
    megatron_model_type: Optional[str]
    model_types: List[str]

    model_provider: Callable[[], nn.Module]
    load_config: Callable[[PretrainedConfig], Dict[str, Any]]
    convert_mcore2hf: Callable[[nn.Module, nn.Module], None]
    convert_hf2mcore: Callable[[nn.Module, nn.Module], None]

    model_groups: List[ModelGroup] = field(default_factory=list)


def register_megatron_model(model_meta: MegatronModelMeta, *, exist_ok: bool = False):
    megatron_model_type = model_meta.megatron_model_type
    for model_type in model_meta.model_types:
        model_meta.model_groups += MODEL_MAPPING[model_type].model_groups
    if not exist_ok and megatron_model_type in MEGATRON_MODEL_MAPPING:
        raise ValueError(f'The `{megatron_model_type}` has already been registered in the MODEL_MAPPING.')

    MEGATRON_MODEL_MAPPING[megatron_model_type] = model_meta


def get_megatron_model_meta(model_id_or_path: str) -> Optional[MegatronModelMeta]:
    return _get_matched_model_meta(model_id_or_path, MEGATRON_MODEL_MAPPING)

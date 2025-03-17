# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import ModelType
from ..config import load_config
from ..constant import MegatronModelType
from ..register import MegatronModelMeta, register_megatron_model
from .hf2mcore import convert_hf2mcore
from .mcore2hf import convert_mcore2hf
from .model import model_provider

register_megatron_model(
    MegatronModelMeta(MegatronModelType.gpt, [
        ModelType.qwen, ModelType.qwen2, ModelType.qwen2_5, ModelType.qwq, ModelType.qwq_preview, ModelType.qwen2_5_math
    ], model_provider, load_config, convert_mcore2hf, convert_hf2mcore))

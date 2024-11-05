# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, Any

from transformers import PretrainedConfig

from swift.llm import TemplateType
from ..constant import LLMModelType
from ..register import (Model, ModelGroup, ModelMeta, register_model, get_model_tokenizer_from_local)


def get_model_tokenizer_telechat(model_dir: str,
                                 config: PretrainedConfig,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    attn_type = AttentionImpl(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    config.flash_attn = attn_type.to_bool()
    return get_model_tokenizer_from_local(
        model_dir, config, model_kwargs, load_model, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.telechat,
        [
            ModelGroup(
                [
                    Model('TeleAI/TeleChat-7B', 'Tele-AI/telechat-7B'),
                    Model('TeleAI/TeleChat-12B', 'Tele-AI/TeleChat-12B'),
                ]),
        ],
        TemplateType.telechat,
        get_model_tokenizer_telechat,
        support_flash_attn=True,
        architectures=['LlavaForConditionalGeneration'],
    ))


register_model(
    ModelMeta(
        LLMModelType.telechat2,
        [
            ModelGroup(
                [
                    Model('TeleAI/TeleChat-12B-v2', 'Tele-AI/TeleChat-12B-v2'),
                ]),
            ModelGroup(
                [
                    Model('swift/TeleChat-12B-V2-GPTQ-Int4'),
                ], requires=['auto_gptq>=0.5']),
        ],
        TemplateType.telechat_v2,
        get_model_tokenizer_telechat,
        support_flash_attn=True,
        architectures=['LlavaForConditionalGeneration'],
    ))

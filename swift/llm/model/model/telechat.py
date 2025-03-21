# Copyright (c) Alibaba, Inc. and its affiliates.

from swift.llm import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model


def get_model_tokenizer_telechat(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    if model is not None:
        for k in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'user_token_id', 'bot_token_id']:
            setattr(tokenizer, k, getattr(model.generation_config, k))
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.telechat,
        [
            ModelGroup([
                Model('TeleAI/TeleChat-7B', 'Tele-AI/telechat-7B'),
                Model('TeleAI/TeleChat-12B', 'Tele-AI/TeleChat-12B'),
                Model('TeleAI/TeleChat-12B-v2', 'Tele-AI/TeleChat-12B-v2'),
                Model('TeleAI/TeleChat-52B', 'TeleAI/TeleChat-52B'),
            ]),
            ModelGroup([
                Model('swift/TeleChat-12B-V2-GPTQ-Int4'),
            ]),
            ModelGroup([
                Model('TeleAI/TeleChat2-35B', 'Tele-AI/TeleChat2-35B'),
                Model('TeleAI/TeleChat2-115B', 'Tele-AI/TeleChat2-115B'),
            ]),
        ],
        TemplateType.telechat,
        get_model_tokenizer_telechat,
        model_arch=ModelArch.telechat,
        architectures=['TelechatForCausalLM', 'TeleChatForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.telechat2,
        [
            ModelGroup([
                Model('TeleAI/TeleChat2-3B', 'Tele-AI/TeleChat2-3B'),
                Model('TeleAI/TeleChat2-7B-32K', 'Tele-AI/TeleChat2-7B-32K'),
                Model('TeleAI/TeleChat2-35B-32K', 'Tele-AI/TeleChat2-35B-32K'),
                Model('TeleAI/TeleChat2-35B-Nov', 'Tele-AI/TeleChat2-35B-Nov'),
            ]),
        ],
        TemplateType.telechat2,
        get_model_tokenizer_telechat,
        model_arch=ModelArch.telechat,
        architectures=['TeleChat2ForCausalLM'],
    ))

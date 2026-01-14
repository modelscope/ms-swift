# Copyright (c) Alibaba, Inc. and its affiliates.

from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

from swift.template import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class TeleChatLoader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, config, processor, **kwargs)
        generation_config = GenerationConfig.from_pretrained(model_dir)
        for k in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'user_token_id', 'bot_token_id']:
            setattr(processor, k, getattr(generation_config, k))
        return model


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
        TeleChatLoader,
        template=TemplateType.telechat,
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
        template=TemplateType.telechat2,
        model_arch=ModelArch.telechat,
        architectures=['TeleChat2ForCausalLM'],
    ))

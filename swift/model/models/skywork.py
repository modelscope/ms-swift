# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from transformers import PretrainedConfig

from swift.template import Processor, TemplateType
from ..constant import LLMModelType, RMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


class SkyworkLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer = super().get_processor(model_dir, config)
        tokenizer.add_tokens('[USER]')
        tokenizer.add_tokens('[BOT]')
        tokenizer.add_tokens('[SEP]')
        return tokenizer


register_model(
    ModelMeta(
        LLMModelType.skywork,
        [
            ModelGroup([
                Model('skywork/Skywork-13B-base', 'skywork/Skywork-13B-base'),
                Model('skywork/Skywork-13B-chat'),
            ]),
        ],
        template=TemplateType.skywork,
        architectures=['SkyworkForCausalLM'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        RMModelType.llama3_2_reward,
        [
            ModelGroup([
                Model('AI-ModelScope/Skywork-Reward-Llama-3.1-8B', 'Skywork/Skywork-Reward-Llama-3.1-8B'),
                Model('AI-ModelScope/Skywork-Reward-Llama-3.1-8B-v0.2', 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/GRM_Llama3.1_8B_rewardmodel-ft', 'Ray2333/GRM_Llama3.1_8B_rewardmodel-ft'),
                Model('AI-ModelScope/GRM-llama3.2-3B-rewardmodel-ft', 'Ray2333/GRM-llama3.2-3B-rewardmodel-ft'),
            ])
        ],
        template=TemplateType.llama3_2,
        requires=['transformers>=4.43'],
        architectures=['LlamaForSequenceClassification'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        RMModelType.gemma_reward,
        [
            ModelGroup([
                Model('AI-ModelScope/Skywork-Reward-Gemma-2-27B', 'Skywork/Skywork-Reward-Gemma-2-27B'),
                Model('AI-ModelScope/Skywork-Reward-Gemma-2-27B-v0.2', 'Skywork/Skywork-Reward-Gemma-2-27B-v0.2'),
            ]),
        ],
        template=TemplateType.gemma,
        requires=['transformers>=4.42'],
        architectures=['Gemma2ForSequenceClassification'],
        model_arch=ModelArch.llama,
    ))

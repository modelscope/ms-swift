# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from swift.llm import TemplateType
from ..constant import LLMModelType, RMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


def get_skywork_model_tokenizer(model_dir: str,
                                model_info,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if 'chat' in model_dir:
        tokenizer.add_tokens('[USER]')
        tokenizer.add_tokens('[BOT]')
        tokenizer.add_tokens('[SEP]')
    return model, tokenizer


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

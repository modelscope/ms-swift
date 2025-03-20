# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo

register_model(
    ModelMeta(
        LLMModelType.mistral,
        [
            ModelGroup([
                Model('AI-ModelScope/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1'),
                Model('AI-ModelScope/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.2'),
                Model('LLM-Research/Mistral-7B-Instruct-v0.3', 'mistralai/Mistral-7B-Instruct-v0.3'),
                Model('AI-ModelScope/Mistral-7B-v0.1', 'mistralai/Mistral-7B-v0.1'),
                Model('AI-ModelScope/Mistral-7B-v0.2-hf', 'alpindale/Mistral-7B-v0.2-hf'),
            ]),
            ModelGroup([
                Model('swift/Codestral-22B-v0.1', 'mistralai/Codestral-22B-v0.1'),
            ]),
        ],
        TemplateType.llama,
        get_model_tokenizer_with_flash_attn,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.34'],
    ))

register_model(
    ModelMeta(
        LLMModelType.mixtral, [
            ModelGroup([
                Model('AI-ModelScope/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
                Model('AI-ModelScope/Mixtral-8x7B-v0.1', 'mistralai/Mixtral-8x7B-v0.1'),
                Model('AI-ModelScope/Mixtral-8x22B-v0.1', 'mistral-community/Mixtral-8x22B-v0.1'),
            ],
                       requires=['transformers>=4.36']),
            ModelGroup([
                Model('AI-ModelScope/Mixtral-8x7b-AQLM-2Bit-1x16-hf', 'ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf'),
            ],
                       requires=['transformers>=4.38', 'aqlm', 'torch>=2.2.0']),
        ],
        TemplateType.llama,
        get_model_tokenizer_with_flash_attn,
        architectures=['MixtralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.mistral_nemo, [
            ModelGroup([
                Model('AI-ModelScope/Mistral-Small-Instruct-2409', 'mistralai/Mistral-Small-Instruct-2409'),
                Model('LLM-Research/Mistral-Large-Instruct-2407', 'mistralai/Mistral-Large-Instruct-2407'),
                Model('AI-ModelScope/Mistral-Nemo-Base-2407', 'mistralai/Mistral-Nemo-Base-2407'),
                Model('AI-ModelScope/Mistral-Nemo-Instruct-2407', 'mistralai/Mistral-Nemo-Instruct-2407'),
            ],
                       requires=['transformers>=4.43']),
            ModelGroup([
                Model('AI-ModelScope/Ministral-8B-Instruct-2410', 'mistralai/Ministral-8B-Instruct-2410'),
            ],
                       requires=['transformers>=4.46']),
        ],
        TemplateType.mistral_nemo,
        get_model_tokenizer_with_flash_attn,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.mistral_2501, [
            ModelGroup([
                Model('mistralai/Mistral-Small-24B-Base-2501', 'mistralai/Mistral-Small-24B-Base-2501'),
                Model('mistralai/Mistral-Small-24B-Instruct-2501', 'mistralai/Mistral-Small-24B-Instruct-2501'),
            ]),
        ],
        TemplateType.mistral_2501,
        get_model_tokenizer_with_flash_attn,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.zephyr,
        [
            ModelGroup([
                Model('modelscope/zephyr-7b-beta', 'HuggingFaceH4/zephyr-7b-beta'),
            ]),
        ],
        TemplateType.zephyr,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MistralForCausalLM'],
        requires=['transformers>=4.34'],
    ))

register_model(
    ModelMeta(
        LLMModelType.wizardlm2_moe,
        [ModelGroup([
            Model('AI-ModelScope/WizardLM-2-8x22B', 'alpindale/WizardLM-2-8x22B'),
        ])],
        TemplateType.wizardlm2_moe,
        get_model_tokenizer_with_flash_attn,
        architectures=['MixtralForCausalLM'],
        requires=['transformers>=4.36'],
    ))

register_model(
    ModelMeta(
        LLMModelType.wizardlm2,
        [ModelGroup([
            Model('AI-ModelScope/WizardLM-2-7B-AWQ', 'MaziyarPanahi/WizardLM-2-7B-AWQ'),
        ])],
        TemplateType.wizardlm2,
        get_model_tokenizer_with_flash_attn,
        architectures=['MistralForCausalLM'],
        requires=['transformers>=4.34'],
    ))


def get_model_tokenizer_mistral_2503(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    try:
        from transformers import Mistral3ForConditionalGeneration
    except ImportError:
        raise ImportError('Please install Gemma3ForConditionalGeneration by running '
                          '`pip install git+https://github.com/huggingface/transformers@v4.49.0-Mistral-3`')

    kwargs['automodel_class'] = kwargs['automodel_class'] or Mistral3ForConditionalGeneration
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.mistral_2503,
        [
            ModelGroup([
                Model('mistralai/Mistral-Small-3.1-24B-Base-2503', 'mistralai/Mistral-Small-3.1-24B-Base-2503'),
                Model('mistralai/Mistral-Small-3.1-24B-Instruct-2503', 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'),
            ]),
        ],
        TemplateType.mistral_2503,
        get_model_tokenizer_mistral_2503,
        architectures=['Mistral3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.49'],
    ), )

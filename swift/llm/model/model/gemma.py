# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from swift.llm import ModelInfo, TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local,
                        get_model_tokenizer_with_flash_attn, register_model)


def get_model_tokenizer_paligemma_vision(model_dir: str,
                                         model_info: ModelInfo,
                                         model_kwargs: Dict[str, Any],
                                         load_model: bool = True,
                                         **kwargs):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_from_local(
        model_dir, model_info, model_kwargs, load_model, automodel_class=PaliGemmaForConditionalGeneration, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.paligemma,
        [
            ModelGroup([
                Model('AI-ModelScope/paligemma-3b-pt-224', 'google/paligemma-3b-pt-224'),
                Model('AI-ModelScope/paligemma-3b-pt-448', 'google/paligemma-3b-pt-448'),
                Model('AI-ModelScope/paligemma-3b-pt-896', 'google/paligemma-3b-pt-896'),
                Model('AI-ModelScope/paligemma-3b-mix-448', 'google/paligemma-3b-mix-448'),
            ],
                       requires=['transformers>=4.41'],
                       tags=['multi-modal', 'vision']),
        ],
        TemplateType.paligemma,
        get_model_tokenizer_paligemma_vision,
        architectures=['PaliGemmaForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma,
        [
            ModelGroup(
                [
                    Model('AI-ModelScope/gemma-2b', 'google/gemma-2b'),
                    Model('AI-ModelScope/gemma-2b-it', 'google/gemma-2b-it'),
                    Model('AI-ModelScope/gemma-7b', 'google/gemma-7b'),
                    Model('AI-ModelScope/gemma-7b-it', 'google/gemma-7b-it'),
                ],
                ignore_file_pattern=[r'.+\.gguf$'],
                requires=['transformers>=4.38'],
            ),
        ],
        TemplateType.gemma,
        get_model_tokenizer_with_flash_attn,
        architectures=['GemmaForCausalLM'],
        support_flash_attn=True,
        support_vllm=True,
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma2,
        [
            ModelGroup(
                [
                    Model('LLM-Research/gemma-2-2b', 'google/gemma-2-2b'),
                    Model('LLM-Research/gemma-2-2b-it', 'google/gemma-2-2b-it'),
                    Model('LLM-Research/gemma-2-9b', 'google/gemma-2-9b'),
                    Model('LLM-Research/gemma-2-9b-it', 'google/gemma-2-9b-it'),
                    Model('LLM-Research/gemma-2-27b', 'google/gemma-2-27b'),
                    Model('LLM-Research/gemma-2-27b-it', 'google/gemma-2-27b-it'),
                ],
                requires=['transformers>=4.42'],
            ),
        ],
        TemplateType.gemma,
        get_model_tokenizer_with_flash_attn,
        architectures=['Gemma2ForCausalLM'],
        support_flash_attn=True,
        support_vllm=True,
    ))

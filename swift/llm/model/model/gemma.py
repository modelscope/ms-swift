# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo


def get_model_tokenizer_paligemma_vision(model_dir: str,
                                         model_info: ModelInfo,
                                         model_kwargs: Dict[str, Any],
                                         load_model: bool = True,
                                         **kwargs):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs.pop('automodel_class', None)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir,
        model_info,
        model_kwargs,
        load_model,
        automodel_class=PaliGemmaForConditionalGeneration,
        tokenizer=processor.tokenizer,
        **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.paligemma, [
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
        model_arch=ModelArch.llava))

register_model(
    ModelMeta(
        LLMModelType.gemma, [
            ModelGroup(
                [
                    Model('AI-ModelScope/gemma-2b-it', 'google/gemma-2b-it'),
                    Model('AI-ModelScope/gemma-2b', 'google/gemma-2b'),
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
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.gemma2, [
            ModelGroup(
                [
                    Model('LLM-Research/gemma-2-2b-it', 'google/gemma-2-2b-it'),
                    Model('LLM-Research/gemma-2-2b', 'google/gemma-2-2b'),
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
        model_arch=ModelArch.llama))

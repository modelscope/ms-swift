# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo


def get_model_tokenizer_paligemma_vision(model_dir: str,
                                         model_info: ModelInfo,
                                         model_kwargs: Dict[str, Any],
                                         load_model: bool = True,
                                         **kwargs):
    from transformers import PaliGemmaForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or PaliGemmaForConditionalGeneration
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.paligemma,
        [
            ModelGroup([
                Model('AI-ModelScope/paligemma-3b-pt-224', 'google/paligemma-3b-pt-224'),
                Model('AI-ModelScope/paligemma-3b-pt-448', 'google/paligemma-3b-pt-448'),
                Model('AI-ModelScope/paligemma-3b-pt-896', 'google/paligemma-3b-pt-896'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma-3b-mix-224', 'google/paligemma-3b-mix-224'),
                Model('AI-ModelScope/paligemma-3b-mix-448', 'google/paligemma-3b-mix-448'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma2-3b-pt-224', 'google/paligemma2-3b-pt-224'),
                Model('AI-ModelScope/paligemma2-3b-pt-448', 'google/paligemma2-3b-pt-448'),
                Model('AI-ModelScope/paligemma2-3b-pt-896', 'google/paligemma2-3b-pt-896'),
                Model('AI-ModelScope/paligemma2-10b-pt-224', 'google/paligemma2-10b-pt-224'),
                Model('AI-ModelScope/paligemma2-10b-pt-448', 'google/paligemma2-10b-pt-448'),
                Model('AI-ModelScope/paligemma2-10b-pt-896', 'google/paligemma2-10b-pt-896'),
                Model('AI-ModelScope/paligemma2-28b-pt-224', 'google/paligemma2-28b-pt-224'),
                Model('AI-ModelScope/paligemma2-28b-pt-448', 'google/paligemma2-28b-pt-448'),
                Model('AI-ModelScope/paligemma2-28b-pt-896', 'google/paligemma2-28b-pt-896'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma2-3b-ft-docci-448', 'google/paligemma2-3b-ft-docci-448'),
                Model('AI-ModelScope/paligemma2-10b-ft-docci-448', 'google/paligemma2-10b-ft-docci-448'),
            ]),
        ],
        TemplateType.paligemma,
        get_model_tokenizer_paligemma_vision,
        architectures=['PaliGemmaForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.41'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma,
        [
            ModelGroup([
                Model('AI-ModelScope/gemma-2b-it', 'google/gemma-2b-it'),
                Model('AI-ModelScope/gemma-2b', 'google/gemma-2b'),
                Model('AI-ModelScope/gemma-7b', 'google/gemma-7b'),
                Model('AI-ModelScope/gemma-7b-it', 'google/gemma-7b-it'),
            ], ),
        ],
        TemplateType.gemma,
        get_model_tokenizer_with_flash_attn,
        architectures=['GemmaForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.38'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma2,
        [
            ModelGroup([
                Model('LLM-Research/gemma-2-2b-it', 'google/gemma-2-2b-it'),
                Model('LLM-Research/gemma-2-2b', 'google/gemma-2-2b'),
                Model('LLM-Research/gemma-2-9b', 'google/gemma-2-9b'),
                Model('LLM-Research/gemma-2-9b-it', 'google/gemma-2-9b-it'),
                Model('LLM-Research/gemma-2-27b', 'google/gemma-2-27b'),
                Model('LLM-Research/gemma-2-27b-it', 'google/gemma-2-27b-it'),
            ], ),
        ],
        TemplateType.gemma,
        get_model_tokenizer_with_flash_attn,
        architectures=['Gemma2ForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.42'],
    ))


def get_model_tokenizer_gemma3_text(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    # It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`.
    kwargs['attn_impl'] = kwargs['attn_impl'] or 'eager'
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.gemma3_text,
        [
            ModelGroup([
                Model('LLM-Research/gemma-3-1b-pt', 'google/gemma-3-1b-pt'),
                Model('LLM-Research/gemma-3-1b-it', 'google/gemma-3-1b-it'),
            ], ),
        ],
        TemplateType.gemma3_text,
        get_model_tokenizer_gemma3_text,
        architectures=['Gemma3ForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.49'],
    ))


def get_model_tokenizer_gemma3_vision(model_dir: str,
                                      model_info: ModelInfo,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    try:
        from transformers import Gemma3ForConditionalGeneration
    except ImportError:
        raise ImportError('Please install Gemma3ForConditionalGeneration by running '
                          '`pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3`')

    kwargs['automodel_class'] = kwargs['automodel_class'] or Gemma3ForConditionalGeneration
    # It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`.
    kwargs['attn_impl'] = kwargs['attn_impl'] or 'eager'
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.gemma3_vision,
        [
            ModelGroup([
                Model('LLM-Research/gemma-3-4b-pt', 'google/gemma-3-4b-pt'),
                Model('LLM-Research/gemma-3-4b-it', 'google/gemma-3-4b-it'),
                Model('LLM-Research/gemma-3-12b-pt', 'google/gemma-3-12b-pt'),
                Model('LLM-Research/gemma-3-12b-it', 'google/gemma-3-12b-it'),
                Model('LLM-Research/gemma-3-27b-pt', 'google/gemma-3-27b-pt'),
                Model('LLM-Research/gemma-3-27b-it', 'google/gemma-3-27b-it'),
            ], ),
        ],
        TemplateType.gemma3_vision,
        get_model_tokenizer_gemma3_vision,
        architectures=['Gemma3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.49'],
    ))

# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType
from typing import Any, Dict, Tuple

import torch
from modelscope import AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, use_submodel_func

logger = get_logger()


def get_model_tokenizer_got_ocr2(*args, **kwargs):
    kwargs['automodel_class'] = AutoModel
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.got_ocr2,
        [
            ModelGroup([
                Model('stepfun-ai/GOT-OCR2_0', 'stepfun-ai/GOT-OCR2_0'),
            ], tags=['audio']),
        ],
        TemplateType.got_ocr2,
        get_model_tokenizer_got_ocr2,
        model_arch=ModelArch.got_ocr2,
        architectures=['GOTQwenForCausalLM'],
    ))


def get_model_tokenizer_idefics(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs['automodel_class'] = AutoModelForVision2Seq
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.idefics3,
        [
            ModelGroup([
                Model('AI-ModelScope/Idefics3-8B-Llama3', 'HuggingFaceM4/Idefics3-8B-Llama3'),
            ],
                       tags=['vision'],
                       requires=['transformers>=4.45']),
        ],
        TemplateType.idefics3,
        get_model_tokenizer_idefics,
        model_arch=ModelArch.idefics3,
        architectures=['Idefics3ForConditionalGeneration'],
    ))


def get_model_tokenizer_pixtral(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs['automodel_class'] = LlavaForConditionalGeneration
    kwargs['tokenizer'] = processor.tokenizer
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.pixtral,
        [
            ModelGroup([
                Model('AI-ModelScope/pixtral-12b', 'mistral-community/pixtral-12b'),
            ]),
        ],
        TemplateType.pixtral,
        get_model_tokenizer_pixtral,
        model_arch=ModelArch.llava_hf,
        architectures=['LlavaForConditionalGeneration'],
        requires=['transformers>=4.45'],
        tags=['vision'],
    ))


def get_model_tokenizer_molmoe(model_dir: str,
                               model_info: ModelInfo,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=processor.tokenizer, **kwargs)

    # fix bug for molmoe-1b
    def to_dict(self, *args, **kwargs):
        res = self._to_dict(*args, **kwargs)
        res['vision_backbone'] = self.vision_backbone.__dict__
        res.pop('to_dict')
        res.pop('_to_dict')
        return res

    model.config._to_dict = model.config.to_dict
    model.config.to_dict = MethodType(to_dict, model.config)
    from transformers import GenerationMixin
    model.generate = MethodType(GenerationMixin.generate, model)

    if model and hasattr(model, '_old_forward'):  # device_map
        device = model.lm_head.weight.device
        forward_origin = model._old_forward

        def _forward(*args, **kwargs):
            if 'append_last_valid_logits' in kwargs:
                kwargs['append_last_valid_logits'] = kwargs['append_last_valid_logits'].to(device)
            return forward_origin(*args, **kwargs)

        model._old_forward = _forward
        model.forward_origin = forward_origin

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.molmoe,
        [
            ModelGroup([
                Model('LLM-Research/MolmoE-1B-0924', 'allenai/MolmoE-1B-0924'),
            ],
                       tags=['vision'],
                       requires=['transformers>=4.45']),
        ],
        TemplateType.molmo,
        get_model_tokenizer_molmoe,
        model_arch=ModelArch.molmo,
        torch_dtype=torch.float32,
        architectures=['MolmoForCausalLM'],
    ))


def get_model_tokenizer_molmo(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module('modeling_molmo.MolmoForCausalLM', model_dir)
    model_cls._no_split_modules = ['MolmoSequentialBlock']
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=processor.tokenizer, **kwargs)
    if model:
        device = next(model.model.transformer.ff_out.parameters()).device
        forward_origin = model.model.forward

        def _forward(*args, **kwargs):
            if 'append_last_valid_logits' in kwargs:
                kwargs['append_last_valid_logits'] = kwargs['append_last_valid_logits'].to(device)
            return forward_origin(*args, **kwargs)

        model.model.forward = _forward
        model.model.forward_origin = forward_origin

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.molmo,
        [
            ModelGroup([
                Model('LLM-Research/Molmo-7B-O-0924', 'allenai/Molmo-7B-O-0924'),
                Model('LLM-Research/Molmo-7B-D-0924', 'allenai/Molmo-7B-D-0924'),
                Model('LLM-Research/Molmo-72B-0924', 'allenai/Molmo-72B-0924'),
            ],
                       tags=['vision'],
                       requires=['transformers>=4.45']),
        ],
        TemplateType.molmo,
        get_model_tokenizer_molmo,
        model_arch=ModelArch.molmo,
        architectures=['MolmoForCausalLM'],
    ))

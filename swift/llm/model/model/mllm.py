# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType
from typing import Any, Dict

import torch
from transformers import AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, use_submodel_func

logger = get_logger()


def get_model_tokenizer_got_ocr2(*args, **kwargs):
    kwargs['automodel_class'] = AutoModel
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.got_ocr2, [
            ModelGroup([
                Model('stepfun-ai/GOT-OCR2_0', 'stepfun-ai/GOT-OCR2_0'),
            ]),
        ],
        TemplateType.got_ocr2,
        get_model_tokenizer_got_ocr2,
        model_arch=ModelArch.got_ocr2,
        architectures=['GOTQwenForCausalLM'],
        tags=['vision']))


def get_model_tokenizer_idefics(model_dir: str, *args, **kwargs):
    from transformers import AutoModelForVision2Seq
    kwargs['automodel_class'] = kwargs['automodel_class'] or AutoModelForVision2Seq
    model, processor = get_model_tokenizer_multimodal(model_dir, *args, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.idefics3,
        [
            ModelGroup([
                Model('AI-ModelScope/Idefics3-8B-Llama3', 'HuggingFaceM4/Idefics3-8B-Llama3'),
            ]),
        ],
        TemplateType.idefics3,
        get_model_tokenizer_idefics,
        model_arch=ModelArch.idefics3,
        architectures=['Idefics3ForConditionalGeneration'],
        tags=['vision'],
        requires=['transformers>=4.45'],
    ))


def get_model_tokenizer_pixtral(model_dir: str, *args, **kwargs):
    from transformers import LlavaForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or LlavaForConditionalGeneration
    model, processor = get_model_tokenizer_multimodal(model_dir, *args, **kwargs)
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
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)

    # fix bug for molmoe-1b
    def to_dict(self, *args, **kwargs):
        res = self._to_dict(*args, **kwargs)
        res['vision_backbone'] = self.vision_backbone.__dict__
        res.pop('to_dict')
        res.pop('_to_dict')
        return res

    if model is not None:
        model.config._to_dict = model.config.to_dict
        model.config.to_dict = MethodType(to_dict, model.config)

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.molmoe,
        [
            ModelGroup([
                Model('LLM-Research/MolmoE-1B-0924', 'allenai/MolmoE-1B-0924'),
            ]),
        ],
        TemplateType.molmo,
        get_model_tokenizer_molmoe,
        model_arch=ModelArch.molmo,
        torch_dtype=torch.float32,
        architectures=['OLMoForCausalLM'],
        tags=['vision'],
        requires=['transformers>=4.45'],
    ))


def get_model_tokenizer_molmo(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    model_cls = get_class_from_dynamic_module('modeling_molmo.MolmoForCausalLM', model_dir)
    model_cls._no_split_modules = ['MolmoSequentialBlock']
    model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)

    patch_output_clone(model.model.transformer.wte)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.molmo,
        [
            ModelGroup([
                Model('LLM-Research/Molmo-7B-O-0924', 'allenai/Molmo-7B-O-0924'),
                Model('LLM-Research/Molmo-7B-D-0924', 'allenai/Molmo-7B-D-0924'),
                Model('LLM-Research/Molmo-72B-0924', 'allenai/Molmo-72B-0924'),
            ]),
        ],
        TemplateType.molmo,
        get_model_tokenizer_molmo,
        model_arch=ModelArch.molmo,
        architectures=['MolmoForCausalLM'],
        tags=['vision'],
        requires=['transformers>=4.45'],
    ))


def get_model_tokenizer_megrez_omni(model_dir, *args, **kwargs):
    model_cls = get_class_from_dynamic_module('modeling_megrezo.MegrezO', model_dir)
    model_cls._no_split_modules = ['ResidualAttentionBlock', 'LlamaDecoderLayer']
    model_cls = get_class_from_dynamic_module('modeling_megrezo.SiglipVisionTransformer', model_dir)
    model_cls._no_split_modules = ['SiglipEncoderLayer']
    model, processor = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    processor = model._get_or_init_processor()
    patch_output_clone(model.llm.model.embed_tokens)
    use_submodel_func(model, 'llm')
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.megrez_omni,
        [
            ModelGroup([
                Model('InfiniAI/Megrez-3B-Omni', 'Infinigence/Megrez-3B-Omni'),
            ]),
        ],
        TemplateType.megrez_omni,
        get_model_tokenizer_megrez_omni,
        model_arch=ModelArch.megrez_omni,
        architectures=['MegrezO'],
        tags=['vision', 'audio'],
    ))

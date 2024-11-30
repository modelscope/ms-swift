# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Dict

from modelscope import AutoConfig, AutoTokenizer, BitsAndBytesConfig, snapshot_download
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, use_submodel_func


def get_model_tokenizer_internlm_chat(model_dir: str,
                                      model_info: ModelInfo,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
        del tokenizer.__class__.eos_token_id
    tokenizer.eos_token = '<eoa>'
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.internlm,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-chat-7b', 'internlm/internlm-chat-7b'),
                Model('Shanghai_AI_Laboratory/internlm-7b', 'internlm/internlm-7b'),
                Model('Shanghai_AI_Laboratory/internlm-chat-7b-8k'),
                Model('Shanghai_AI_Laboratory/internlm-20b', 'internlm/internlm-20b'),
                Model('Shanghai_AI_Laboratory/internlm-chat-20b', 'internlm/internlm-chat-20b'),
            ])
        ],
        TemplateType.internlm,
        get_model_tokenizer_internlm_chat,
        architectures=['InternLMForCausalLM'],
        model_arch=ModelArch.llama,
    ))


def get_model_tokenizer_internlm2(model_dir: str,
                                  model_info: ModelInfo,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    eos_token = kwargs.pop('eos_token', None)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if eos_token is not None:
        if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
            del tokenizer.__class__.eos_token_id
        tokenizer.eos_token = eos_token

    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.internlm2,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm2-chat-1_8b', 'internlm/internlm2-chat-1_8b'),
                Model('Shanghai_AI_Laboratory/internlm2-1_8b', 'internlm/internlm2-1_8b'),
                Model('Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft', 'internlm/internlm2-chat-1_8b-sft'),
                Model('Shanghai_AI_Laboratory/internlm2-base-7b', 'internlm/internlm2-base-7b'),
                Model('Shanghai_AI_Laboratory/internlm2-7b', 'internlm/internlm2-7b'),
                Model('Shanghai_AI_Laboratory/internlm2-chat-7b', 'internlm/internlm2-chat-7b'),
                Model('Shanghai_AI_Laboratory/internlm2-chat-7b-sft', 'internlm/internlm2-chat-7b-sft'),
                Model('Shanghai_AI_Laboratory/internlm2-base-20b', 'internlm/internlm2-base-20b'),
                Model('Shanghai_AI_Laboratory/internlm2-20b', 'internlm/internlm2-20b'),
                Model('Shanghai_AI_Laboratory/internlm2-chat-20b', 'internlm/internlm2-chat-20b'),
                Model('Shanghai_AI_Laboratory/internlm2-chat-20b-sft', 'internlm/internlm2-chat-20b-sft'),
            ]),
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm2-math-7b', 'internlm/internlm2-math-7b'),
                Model('Shanghai_AI_Laboratory/internlm2-math-base-7b', 'internlm/internlm2-math-base-7b'),
                Model('Shanghai_AI_Laboratory/internlm2-math-base-20b', 'internlm/internlm2-math-base-20b'),
                Model('Shanghai_AI_Laboratory/internlm2-math-20b', 'internlm/internlm2-math-20b'),
            ],
                       tags=['math']),
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm2_5-1_8b-chat', 'internlm/internlm2_5-1_8b-chat'),
                Model('Shanghai_AI_Laboratory/internlm2_5-1_8b', 'internlm/internlm2_5-1_8b'),
                Model('Shanghai_AI_Laboratory/internlm2_5-7b', 'internlm/internlm2_5-7b'),
                Model('Shanghai_AI_Laboratory/internlm2_5-7b-chat', 'internlm/internlm2_5-7b-chat'),
                Model('Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m', 'internlm/internlm2_5-7b-chat-1m'),
                Model('Shanghai_AI_Laboratory/internlm2_5-20b', 'internlm/internlm2_5-20b'),
                Model('Shanghai_AI_Laboratory/internlm2_5-20b-chat', 'internlm/internlm2_5-20b-chat'),
            ],
                       tags=['math'])
        ],
        TemplateType.internlm2,
        get_model_tokenizer_internlm2,
        requires=['transformers>=4.38'],
        architectures=['InternLM2ForCausalLM'],
        model_arch=ModelArch.internlm2,
    ))


def get_model_tokenizer_internlm_xcomposer2(model_dir: str,
                                            model_info: ModelInfo,
                                            model_kwargs: Dict[str, Any],
                                            load_model: bool = True,
                                            **kwargs):
    version = kwargs.pop('version', 'v2')
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if version == 'v2-4khd':
        from transformers import CLIPVisionModel

        def load_model(self):
            self.vision_tower_name = snapshot_download('AI-ModelScope/clip-vit-large-patch14-336')
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)
            self.is_loaded = True

        CLIPVisionTower = get_class_from_dynamic_module('build_mlp.CLIPVisionTower', model_dir)
        CLIPVisionTower.load_model = load_model

    model, tokenizer = get_model_tokenizer_internlm2(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None:
        if version == 'v2' and use_flash_attn:
            # fix AttributeError: no attribute 'attention_dropout'
            model.model.layers[0].attention.__class__.attention_dropout = 0.

        if version == 'v2.5':
            patch_output_to_input_device(model.vit)
            patch_output_to_input_device(model.vision_proj)

    return model, tokenizer


def get_model_tokenizer_internvl(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = kwargs.get('model_config')
    if not model_config:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    if kwargs.get('eos_token') is None and tokenizer.eos_token != '<|im_end|>':
        try:
            del tokenizer.__class__.eos_token_id
        except AttributeError:
            pass
        tokenizer.eos_token = '<|im_end|>'

    model_quant_config = getattr(model_config, 'quantization_config', None)

    use_bnb = False
    if model_quant_config is not None:
        use_bnb = model_quant_config.get('quant_method', None) == 'bitsandbytes'
    quantization_config = model_kwargs.get('quantization_config', None)
    if isinstance(quantization_config, BitsAndBytesConfig):
        use_bnb = True

    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)

    if use_bnb and kwargs.get('is_training'):
        # patch: bnb backward shape mismatch bug
        if model is not None and model.language_model is not None:
            model.language_model.output.state.force_no_igemmlt = True

    if model is not None:
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        use_submodel_func(model, 'language_model', func_list)
        patch_output_clone(model.language_model.get_input_embeddings())

    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.internvl,
        [
            ModelGroup(
                [
                    Model('AI-ModelScope/InternVL-Chat-V1-5', 'OpenGVLab/InternVL-Chat-V1-5'),
                    Model('AI-ModelScope/InternVL-Chat-V1-5-int8', 'OpenGVLab/InternVL-Chat-V1-5-int8'),
                    Model('OpenGVLab/Mini-InternVL-Chat-2B-V1-5', 'OpenGVLab/Mini-InternVL-Chat-2B-V1-5'),
                ],
                requires=['transformers>=4.35', 'timm'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.internvl,
        get_model_tokenizer_internvl,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.internvl,
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl_mini,
        [
            ModelGroup(
                [
                    Model('OpenGVLab/Mini-InternVL-Chat-4B-V1-5', 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'),
                ],
                requires=['transformers>=4.35,<4.42', 'timm'],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.internvl_phi3,
        get_model_tokenizer_internvl,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.internvl,
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl2,
        [
            ModelGroup(
                [
                    Model('OpenGVLab/InternVL2-1B', 'OpenGVLab/InternVL2-1B'),
                    Model('OpenGVLab/InternVL2-2B', 'OpenGVLab/InternVL2-2B'),
                    Model('OpenGVLab/InternVL2-8B', 'OpenGVLab/InternVL2-8B'),
                    Model('OpenGVLab/InternVL2-26B', 'OpenGVLab/InternVL2-26B'),
                    Model('OpenGVLab/InternVL2-40B', 'OpenGVLab/InternVL2-40B'),
                    Model('OpenGVLab/InternVL2-Llama3-76B', 'OpenGVLab/InternVL2-Llama3-76B'),
                    Model('OpenGVLab/InternVL2-2B-AWQ', 'OpenGVLab/InternVL2-2B-AWQ'),
                    Model('OpenGVLab/InternVL2-8B-AWQ', 'OpenGVLab/InternVL2-8B-AWQ'),
                    Model('OpenGVLab/InternVL2-26B-AWQ', 'OpenGVLab/InternVL2-26B-AWQ'),
                    Model('OpenGVLab/InternVL2-40B-AWQ', 'OpenGVLab/InternVL2-40B-AWQ'),
                    Model('OpenGVLab/InternVL2-Llama3-76B-AWQ', 'OpenGVLab/InternVL2-Llama3-76B-AWQ'),
                ],
                requires=['transformers>=4.36', 'timm'],
                tags=['multi-modal', 'vision', 'video'],
                ignore_file_pattern=[r'.+\.zip$'],
            ),
        ],
        TemplateType.internvl2,
        get_model_tokenizer_internvl,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.internvl,
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl2_phi3,
        [
            ModelGroup(
                [
                    Model('OpenGVLab/InternVL2-4B', 'OpenGVLab/InternVL2-4B'),
                ],
                requires=['transformers>=4.36,<4.42', 'timm'],
                tags=['multi-modal', 'vision', 'video'],
                ignore_file_pattern=[r'.+\.zip$'],
            ),
        ],
        TemplateType.internvl2_phi3,
        get_model_tokenizer_internvl,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.internvl,
    ))

register_model(
    ModelMeta(
        MLLMModelType.xcomposer2_5, [
            ModelGroup(
                [
                    Model('Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b', 'internlm/internlm-xcomposer2d5-7b'),
                ],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.xcomposer2_5,
        get_model_tokenizer_internlm_xcomposer2,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.internlm_xcomposer))

register_model(
    ModelMeta(
        MLLMModelType.xcomposer2, [
            ModelGroup(
                [
                    Model('Shanghai_AI_Laboratory/internlm-xcomposer2-7b', 'internlm/internlm-xcomposer2-7b'),
                ],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.xcomposer2,
        get_model_tokenizer_internlm_xcomposer2,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.internlm_xcomposer))

register_model(
    ModelMeta(
        MLLMModelType.xcomposer2_4khd, [
            ModelGroup(
                [
                    Model('Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b', 'internlm/internlm-xcomposer2-4khd-7b'),
                ],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.xcomposer2,
        partial(get_model_tokenizer_internlm_xcomposer2, version='v2-4khd'),
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.internlm_xcomposer))

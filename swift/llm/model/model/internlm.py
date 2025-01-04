# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Dict

from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType, RMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_reward_model,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, safe_snapshot_download, use_submodel_func

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
        get_model_tokenizer_with_flash_attn,
        architectures=['InternLMForCausalLM'],
        model_arch=ModelArch.llama,
    ))

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
            ])
        ],
        TemplateType.internlm2,
        get_model_tokenizer_with_flash_attn,
        requires=['transformers>=4.38'],
        architectures=['InternLM2ForCausalLM'],
        model_arch=ModelArch.internlm2,
    ))


def get_model_tokenizer_xcomposer2(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    version = kwargs.pop('version', 'v2')
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if version == 'v2-4khd':
        from transformers import CLIPVisionModel

        def load_model(self):
            self.vision_tower_name = safe_snapshot_download('AI-ModelScope/clip-vit-large-patch14-336')
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)
            self.is_loaded = True

        CLIPVisionTower = get_class_from_dynamic_module('build_mlp.CLIPVisionTower', model_dir)
        CLIPVisionTower.load_model = load_model

    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    model.vit.vision_tower.gradient_checkpointing_enable()
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
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if model_info.quant_method == 'bnb' and kwargs.get('is_training'):
        # patch: bnb backward shape mismatch bug
        if model is not None and model.language_model is not None:
            model.language_model.output.state.force_no_igemmlt = True

    if model is not None:
        use_submodel_func(model, 'language_model')
        patch_output_clone(model.language_model.get_input_embeddings())

    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.internvl,
        [
            ModelGroup([
                Model('OpenGVLab/Mini-InternVL-Chat-2B-V1-5', 'OpenGVLab/Mini-InternVL-Chat-2B-V1-5'),
                Model('AI-ModelScope/InternVL-Chat-V1-5', 'OpenGVLab/InternVL-Chat-V1-5'),
                Model('AI-ModelScope/InternVL-Chat-V1-5-int8', 'OpenGVLab/InternVL-Chat-V1-5-int8'),
            ], ),
        ],
        TemplateType.internvl,
        get_model_tokenizer_internvl,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.35', 'timm'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl_phi3,
        [
            ModelGroup([
                Model('OpenGVLab/Mini-InternVL-Chat-4B-V1-5', 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'),
            ], ),
        ],
        TemplateType.internvl_phi3,
        get_model_tokenizer_internvl,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.35,<4.42', 'timm'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl2,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL2-1B', 'OpenGVLab/InternVL2-1B'),
                Model('OpenGVLab/InternVL2-2B', 'OpenGVLab/InternVL2-2B'),
                Model('OpenGVLab/InternVL2-8B', 'OpenGVLab/InternVL2-8B'),
                Model('OpenGVLab/InternVL2-26B', 'OpenGVLab/InternVL2-26B'),
                Model('OpenGVLab/InternVL2-40B', 'OpenGVLab/InternVL2-40B'),
                Model('OpenGVLab/InternVL2-Llama3-76B', 'OpenGVLab/InternVL2-Llama3-76B'),
            ]),
            # (infer use lmdeploy)
            ModelGroup([
                Model('OpenGVLab/InternVL2-2B-AWQ', 'OpenGVLab/InternVL2-2B-AWQ'),
                Model('OpenGVLab/InternVL2-8B-AWQ', 'OpenGVLab/InternVL2-8B-AWQ'),
                Model('OpenGVLab/InternVL2-26B-AWQ', 'OpenGVLab/InternVL2-26B-AWQ'),
                Model('OpenGVLab/InternVL2-40B-AWQ', 'OpenGVLab/InternVL2-40B-AWQ'),
                Model('OpenGVLab/InternVL2-Llama3-76B-AWQ', 'OpenGVLab/InternVL2-Llama3-76B-AWQ'),
            ]),
            ModelGroup([Model('OpenGVLab/InternVL2-8B-MPO', 'OpenGVLab/InternVL2-8B-MPO')]),
            # pretrain
            ModelGroup([
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-1B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-1B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-2B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-2B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-4B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-4B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-8B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-8B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-26B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-26B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-40B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-40B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-Llama3-76B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-Llama3-76B-Pretrain'),
            ])
        ],
        TemplateType.internvl2,
        get_model_tokenizer_internvl,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.36', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl2_phi3,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL2-4B', 'OpenGVLab/InternVL2-4B'),
            ], ),
        ],
        TemplateType.internvl2_phi3,
        get_model_tokenizer_internvl,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.36,<4.42', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl2_5,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL2_5-1B', 'OpenGVLab/InternVL2_5-1B'),
                Model('OpenGVLab/InternVL2_5-2B', 'OpenGVLab/InternVL2_5-2B'),
                Model('OpenGVLab/InternVL2_5-4B', 'OpenGVLab/InternVL2_5-4B'),
                Model('OpenGVLab/InternVL2_5-8B', 'OpenGVLab/InternVL2_5-8B'),
                Model('OpenGVLab/InternVL2_5-26B', 'OpenGVLab/InternVL2_5-26B'),
                Model('OpenGVLab/InternVL2_5-38B', 'OpenGVLab/InternVL2_5-38B'),
                Model('OpenGVLab/InternVL2_5-78B', 'OpenGVLab/InternVL2_5-78B'),
            ]),
            # quant (infer use lmdeploy)
            ModelGroup([
                Model('OpenGVLab/InternVL2_5-4B-AWQ', 'OpenGVLab/InternVL2_5-4B-AWQ'),
                Model('OpenGVLab/InternVL2_5-8B-AWQ', 'OpenGVLab/InternVL2_5-8B-AWQ'),
                Model('OpenGVLab/InternVL2_5-26B-AWQ', 'OpenGVLab/InternVL2_5-26B-AWQ'),
                Model('OpenGVLab/InternVL2_5-38B-AWQ', 'OpenGVLab/InternVL2_5-38B-AWQ'),
                Model('OpenGVLab/InternVL2_5-78B-AWQ', 'OpenGVLab/InternVL2_5-78B-AWQ'),
            ]),
            ModelGroup([
                Model('OpenGVLab/InternVL2_5-1B-MPO', 'OpenGVLab/InternVL2_5-1B-MPO'),
                Model('OpenGVLab/InternVL2_5-2B-MPO', 'OpenGVLab/InternVL2_5-2B-MPO'),
                Model('OpenGVLab/InternVL2_5-4B-MPO', 'OpenGVLab/InternVL2_5-4B-MPO'),
                Model('OpenGVLab/InternVL2_5-8B-MPO', 'OpenGVLab/InternVL2_5-8B-MPO'),
                Model('OpenGVLab/InternVL2_5-26B-MPO', 'OpenGVLab/InternVL2_5-26B-MPO'),
                Model('OpenGVLab/InternVL2_5-38B-MPO', 'OpenGVLab/InternVL2_5-38B-MPO'),
                Model('OpenGVLab/InternVL2_5-78B-MPO', 'OpenGVLab/InternVL2_5-78B-MPO'),
            ])
        ],
        TemplateType.internvl2_5,
        get_model_tokenizer_internvl,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.36', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.xcomposer2_5,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b', 'internlm/internlm-xcomposer2d5-7b'),
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b:base',
                      'internlm/internlm-xcomposer2d5-ol-7b:base')
            ]),
        ],
        TemplateType.xcomposer2_5,
        partial(get_model_tokenizer_xcomposer2, version='v2.5'),
        architectures=['InternLMXComposer2ForCausalLM'],
        model_arch=ModelArch.xcomposer,
        tags=['vision'],
        requires=['decord'],
        # target_modules: attention.wqkv attention.wo feed_forward.w1 feed_forward.w2 feed_forward.w3
    ))

register_model(
    ModelMeta(
        MLLMModelType.xcomposer2,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2-7b', 'internlm/internlm-xcomposer2-7b'),
            ], ),
        ],
        TemplateType.xcomposer2,
        get_model_tokenizer_xcomposer2,
        architectures=['InternLMXComposer2ForCausalLM'],
        model_arch=ModelArch.xcomposer,
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.xcomposer2_4khd,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b', 'internlm/internlm-xcomposer2-4khd-7b'),
            ], ),
        ],
        TemplateType.xcomposer2,
        partial(get_model_tokenizer_xcomposer2, version='v2-4khd'),
        architectures=['InternLM2ForCausalLM', 'InternLMXComposer2ForCausalLM'],
        model_arch=ModelArch.xcomposer,
        tags=['vision'],
    ))


def get_model_tokenizer_xcomposer_ol(model_dir, *args, **kwargs):
    model_tag = model_dir.rsplit('/', 1)[-1]
    if model_tag == 'audio':
        from .qwen import get_model_tokenizer_qwen2_audio
        return get_model_tokenizer_qwen2_audio(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.xcomposer2_5_ol_audio,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b:audio',
                      'internlm/internlm-xcomposer2d5-ol-7b:audio'),
            ]),
        ],
        TemplateType.qwen2_audio,
        get_model_tokenizer_xcomposer_ol,
        requires=['transformers>=4.45'],
        architectures=['Qwen2AudioForConditionalGeneration'],
        model_arch=ModelArch.qwen2_audio,
        tags=['audio'],
    ))

register_model(
    ModelMeta(
        RMModelType.internlm2_reward,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm2-1_8b-reward', 'internlm/internlm2-1_8b-reward'),
                Model('Shanghai_AI_Laboratory/internlm2-7b-reward', 'internlm/internlm2-7b-reward'),
                Model('Shanghai_AI_Laboratory/internlm2-20b-reward', 'internlm/internlm2-20b-reward'),
            ]),
        ],
        TemplateType.internlm2_reward,
        get_model_tokenizer_reward_model,
        requires=['transformers>=4.38'],
        architectures=['InternLM2ForRewardModel'],
    ))

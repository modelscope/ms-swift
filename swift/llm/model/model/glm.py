# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
from typing import Any, Dict, Type

import torch
import transformers
from packaging import version
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.llm import TemplateType
from swift.utils import get_device_count, get_dist_setting, get_logger
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_get_input_embeddings, patch_output_to_input_device
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import AttnImpl, ModelInfo, safe_snapshot_download

logger = get_logger()


def remove_property(tokenizer_cls: Type[PreTrainedTokenizerBase], tokenizer_config: Dict[str, Any]) -> None:
    for k, v in tokenizer_cls.__dict__.items():
        if k.endswith('_token') and isinstance(v, property) and k in tokenizer_config:
            setattr(tokenizer_cls, k, tokenizer_config[k])


def _patch_tokenizer(tokenizer):
    tokenizer_cls = tokenizer.__class__
    if hasattr(tokenizer_cls, '_origin_pad'):
        return
    tokenizer_cls._origin_pad = tokenizer_cls._pad
    parameters = inspect.signature(tokenizer_cls._origin_pad).parameters

    def _pad(self, *args, **kwargs):
        if 'padding_side' in kwargs and kwargs['padding_side'] is None and 'padding_side' not in parameters:
            kwargs.pop('padding_side')
        return tokenizer_cls._origin_pad(self, *args, **kwargs)

    tokenizer_cls._pad = _pad


def get_model_tokenizer_chatglm(model_dir: str,
                                model_info: ModelInfo,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    if model_kwargs.get('quantization_config') is not None:
        model_kwargs['quantization_config'].llm_int8_skip_modules = ['output_layer']
    # fix transformers>=4.34 bug
    if version.parse(transformers.__version__) >= version.parse('4.34'):
        tokenizer_config = get_tokenizer_config(model_dir)
        class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
        tokenizer_cls: Type[PreTrainedTokenizerBase] = get_class_from_dynamic_module(class_ref, model_dir)
        tokenizer_cls._auto_class = 'AutoTokenizer'
        remove_property(tokenizer_cls, tokenizer_config)
        kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    _patch_tokenizer(tokenizer)
    if model is not None:
        from torch.nn import CrossEntropyLoss
        __old_forward = CrossEntropyLoss.forward

        def cross_entropy_forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            target = target.to(device=inputs.device)
            return __old_forward(self, inputs, target)

        CrossEntropyLoss.forward = cross_entropy_forward

    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.chatglm2, [
            ModelGroup([
                Model('ZhipuAI/chatglm2-6b', 'zai-org/chatglm2-6b'),
                Model('ZhipuAI/chatglm2-6b-32k', 'zai-org/chatglm2-6b-32k')
            ],
                       requires=['transformers<4.42']),
            ModelGroup(
                [Model('ZhipuAI/codegeex2-6b', 'zai-org/codegeex2-6b')],
                requires=['transformers<4.34'],
                tags=['coding'],
            ),
        ],
        TemplateType.chatglm2,
        get_model_tokenizer_chatglm,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.chatglm))

register_model(
    ModelMeta(
        LLMModelType.chatglm3, [
            ModelGroup([
                Model('ZhipuAI/chatglm3-6b', 'zai-org/chatglm3-6b'),
                Model('ZhipuAI/chatglm3-6b-base', 'zai-org/chatglm3-6b-base'),
                Model('ZhipuAI/chatglm3-6b-32k', 'zai-org/chatglm3-6b-32k'),
                Model('ZhipuAI/chatglm3-6b-128k', 'zai-org/chatglm3-6b-128k'),
            ])
        ],
        TemplateType.glm4,
        get_model_tokenizer_chatglm,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.chatglm))


def get_model_tokenizer_glm4(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'))
    kwargs['model_config'] = model_config
    model, tokenizer = get_model_tokenizer_chatglm(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if len(tokenizer.encode('<|user|>', add_special_tokens=False)) > 1:
        for k in tokenizer.special_tokens.keys():
            tokenizer.add_tokens(k)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.glm4,
        [
            ModelGroup([
                Model('ZhipuAI/glm-4-9b-chat', 'zai-org/glm-4-9b-chat'),
                Model('ZhipuAI/glm-4-9b', 'zai-org/glm-4-9b'),
                Model('ZhipuAI/glm-4-9b-chat-1m', 'zai-org/glm-4-9b-chat-1m'),
            ]),
            ModelGroup([
                Model('ZhipuAI/LongWriter-glm4-9b', 'zai-org/LongWriter-glm4-9b'),
            ])
        ],
        TemplateType.glm4,
        get_model_tokenizer_glm4,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.chatglm,
        requires=['transformers>=4.42'],
    ))

register_model(
    ModelMeta(
        LLMModelType.glm4_0414,
        [
            ModelGroup([
                Model('ZhipuAI/GLM-4-9B-0414', 'zai-org/GLM-4-9B-0414'),
                Model('ZhipuAI/GLM-4-32B-0414', 'zai-org/GLM-4-32B-0414'),
                Model('ZhipuAI/GLM-4-32B-Base-0414', 'zai-org/GLM-4-32B-Base-0414'),
                Model('ZhipuAI/GLM-Z1-9B-0414', 'zai-org/GLM-Z1-9B-0414'),
                Model('ZhipuAI/GLM-Z1-32B-0414', 'zai-org/GLM-Z1-32B-0414'),
            ])
        ],
        TemplateType.glm4_0414,
        get_model_tokenizer_with_flash_attn,
        architectures=['Glm4ForCausalLM'],
        model_arch=ModelArch.chatglm,
        requires=['transformers>=4.51'],
    ))

register_model(
    ModelMeta(
        LLMModelType.glm4_z1_rumination,
        [ModelGroup([
            Model('ZhipuAI/GLM-Z1-Rumination-32B-0414', 'zai-org/GLM-Z1-Rumination-32B-0414'),
        ])],
        TemplateType.glm4_z1_rumination,
        get_model_tokenizer_with_flash_attn,
        architectures=['Glm4ForCausalLM'],
        model_arch=ModelArch.chatglm,
        requires=['transformers>4.51'],
    ))

register_model(
    ModelMeta(
        LLMModelType.longwriter_llama3_1,
        [ModelGroup([
            Model('ZhipuAI/LongWriter-llama3.1-8b', 'zai-org/LongWriter-llama3.1-8b'),
        ])],
        TemplateType.longwriter_llama,
        get_model_tokenizer_with_flash_attn,
        architectures=['LlamaForCausalLM'],
        requires=['transformers>=4.43'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        LLMModelType.codegeex4,
        [ModelGroup([
            Model('ZhipuAI/codegeex4-all-9b', 'zai-org/codegeex4-all-9b'),
        ])],
        TemplateType.codegeex4,
        get_model_tokenizer_glm4,
        requires=['transformers<4.42'],
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.chatglm,
        tags=['coding'],
    ))


def get_model_tokenizer_glm4v(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    model, tokenizer = get_model_tokenizer_glm4(model_dir, model_info, model_kwargs, load_model, **kwargs)
    # fix merge-lora
    tokenizer.init_kwargs['image_size'] = 1120
    if load_model:
        # fix device_map 4
        n_gpu = get_device_count()
        local_world_size = get_dist_setting()[3]
        if n_gpu // local_world_size >= 4:
            for layer in model.transformer.vision.transformer.layers:
                patch_output_to_input_device(layer.mlp)
                patch_output_to_input_device(layer.post_attention_layernorm)
            device = next(model.transformer.vision.linear_proj.parameters()).device
            model.transformer.vision.boi.data = model.transformer.vision.boi.to(device)
            model.transformer.vision.eoi.data = model.transformer.vision.eoi.to(device)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.glm4v,
        [
            ModelGroup(
                [
                    Model('ZhipuAI/glm-4v-9b', 'zai-org/glm-4v-9b'),
                ],
                requires=['transformers>=4.42,<4.45'],
            ),
            ModelGroup(
                [
                    Model('ZhipuAI/cogagent-9b-20241220', 'zai-org/cogagent-9b-20241220'),
                ],
                requires=['transformers>=4.42'],
            )
        ],
        TemplateType.glm4v,
        get_model_tokenizer_glm4v,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.glm4v,
    ))


def get_model_tokenizer_glm4_1v(*args, **kwargs):
    from transformers import Glm4vForConditionalGeneration
    logger.info(
        "If you encounter the error 'TypeError: group_images_by_shape() missing 1 required positional argument: "
        "\"disable_grouping\"', please install the source version of the transformers library.")

    kwargs['automodel_class'] = kwargs['automodel_class'] or Glm4vForConditionalGeneration
    model, processor = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None and hasattr(model, 'visual'):
        patch_get_input_embeddings(model.visual, 'patch_embed')
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.glm4_1v,
        [
            ModelGroup(
                [
                    Model('ZhipuAI/GLM-4.1V-9B-Base', 'zai-org/GLM-4.1V-9B-Base'),
                    Model('ZhipuAI/GLM-4.1V-9B-Thinking', 'zai-org/GLM-4.1V-9B-Thinking'),
                ],
                requires=['transformers>=4.53'],
            ),
        ],
        TemplateType.glm4_1v,
        get_model_tokenizer_glm4_1v,
        architectures=['Glm4vForConditionalGeneration'],
        model_arch=ModelArch.glm4_1v,
    ))


def get_model_tokenizer_cogvlm(model_dir: str,
                               model_info: ModelInfo,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    tokenizer_dir = safe_snapshot_download('AI-ModelScope/vicuna-7b-v1.5', download_model=False, check_local=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if load_model:
        logger.warning('CogAgent with FusedLayerNorm will cause an training loss of NAN, '
                       'to avoid this, please uninstall apex.')
        logger.info('Please ignore the unimported warning.')
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.cogvlm, [
            ModelGroup([
                Model('ZhipuAI/cogvlm-chat', 'zai-org/cogvlm-chat-hf'),
            ]),
        ],
        TemplateType.cogvlm,
        get_model_tokenizer_cogvlm,
        architectures=['CogVLMForCausalLM'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.cogvlm))

register_model(
    ModelMeta(
        MLLMModelType.cogagent_chat, [
            ModelGroup([
                Model('ZhipuAI/cogagent-chat', 'zai-org/cogagent-chat-hf'),
            ]),
        ],
        TemplateType.cogagent_chat,
        get_model_tokenizer_cogvlm,
        architectures=['CogAgentForCausalLM'],
        requires=['transformers<4.42', 'timm'],
        model_arch=ModelArch.cogvlm))

register_model(
    ModelMeta(
        MLLMModelType.cogagent_vqa, [ModelGroup([
            Model('ZhipuAI/cogagent-vqa', 'zai-org/cogagent-vqa-hf'),
        ])],
        TemplateType.cogagent_vqa,
        get_model_tokenizer_cogvlm,
        architectures=['CogAgentForCausalLM'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.cogvlm))


def get_model_tokenizer_cogvlm2(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    if model is not None:
        # fix device map 4
        for layer in model.model.vision.transformer.layers:
            patch_output_to_input_device(layer.mlp)
            patch_output_to_input_device(layer.post_attention_layernorm)

        device = next(model.model.vision.linear_proj.parameters()).device
        model.model.vision.boi.data = model.model.vision.boi.to(device)
        model.model.vision.eoi.data = model.model.vision.eoi.to(device)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.cogvlm2, [
            ModelGroup([
                Model('ZhipuAI/cogvlm2-llama3-chat-19B', 'zai-org/cogvlm2-llama3-chat-19B'),
                Model('ZhipuAI/cogvlm2-llama3-chinese-chat-19B', 'zai-org/cogvlm2-llama3-chinese-chat-19B'),
            ]),
        ],
        TemplateType.cogvlm2,
        get_model_tokenizer_cogvlm2,
        architectures=['CogVLMForCausalLM'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.cogvlm))

register_model(
    ModelMeta(
        MLLMModelType.cogvlm2_video,
        [
            ModelGroup([
                Model('ZhipuAI/cogvlm2-video-llama3-chat', 'zai-org/cogvlm2-video-llama3-chat'),
            ]),
        ],
        TemplateType.cogvlm2_video,
        get_model_tokenizer_cogvlm2,
        architectures=['CogVLMVideoForCausalLM'],
        requires=['decord', 'pytorchvideo', 'transformers>=4.42'],
        model_arch=ModelArch.cogvlm,
        tags=['video'],
    ))

register_model(
    ModelMeta(
        LLMModelType.glm_edge,
        [
            ModelGroup([
                Model('ZhipuAI/glm-edge-1.5b-chat', 'zai-org/glm-edge-1.5b-chat'),
                Model('ZhipuAI/glm-edge-4b-chat', 'zai-org/glm-edge-4b-chat'),
            ]),
        ],
        TemplateType.glm4,
        get_model_tokenizer_with_flash_attn,
        architectures=['GlmForCausalLM'],
        requires=['transformers>=4.46'],
    ))


def get_model_tokenizer_glm_edge_v(model_dir: str, *args, **kwargs):
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    processor.tokenizer = tokenizer
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.glm_edge_v,
        [
            ModelGroup([
                Model('ZhipuAI/glm-edge-v-2b', 'zai-org/glm-edge-v-2b'),
                Model('ZhipuAI/glm-edge-4b-chat', 'zai-org/glm-edge-4b-chat'),
            ]),
        ],
        TemplateType.glm_edge_v,
        get_model_tokenizer_glm_edge_v,
        architectures=['GlmForCausalLM'],
        requires=['transformers>=4.46'],
        model_arch=ModelArch.glm_edge_v,
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        LLMModelType.glm4_5,
        [
            ModelGroup([
                Model('ZhipuAI/GLM-4.5-Air-Base', 'zai-org/GLM-4.5-Air-Base'),
                Model('ZhipuAI/GLM-4.5-Air', 'zai-org/GLM-4.5-Air'),
                Model('ZhipuAI/GLM-4.5-Air-FP8', 'zai-org/GLM-4.5-Air-FP8'),
                Model('ZhipuAI/GLM-4.5-Base', 'zai-org/GLM-4.5-Base'),
                Model('ZhipuAI/GLM-4.5', 'zai-org/GLM-4.5'),
                Model('ZhipuAI/GLM-4.5-FP8', 'zai-org/GLM-4.5-FP8'),
            ]),
            ModelGroup([
                Model('ZhipuAI/GLM-4.6', 'zai-org/GLM-4.6'),
            ])
        ],
        TemplateType.glm4_5,
        get_model_tokenizer_with_flash_attn,
        architectures=['Glm4MoeForCausalLM'],
        requires=['transformers>=4.54'],
    ))


def get_model_tokenizer_glm4_5v(*args, **kwargs):
    from transformers import Glm4vMoeForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Glm4vMoeForConditionalGeneration
    model, processor = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None:
        patch_get_input_embeddings(model.visual, 'patch_embed')
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.glm4_5v,
        [
            ModelGroup([
                Model('ZhipuAI/GLM-4.5V', 'zai-org/GLM-4.5V'),
                Model('ZhipuAI/GLM-4.5V-FP8', 'zai-org/GLM-4.5V-FP8'),
            ]),
        ],
        TemplateType.glm4_5v,
        get_model_tokenizer_glm4_5v,
        model_arch=ModelArch.glm4_1v,
        architectures=['Glm4vMoeForConditionalGeneration'],
        requires=['transformers>=4.56.0.dev'],
    ))

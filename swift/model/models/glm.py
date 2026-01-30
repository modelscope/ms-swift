# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
from typing import Any, Dict, Type

import torch
import transformers
from packaging import version
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.template import TemplateType
from swift.utils import Processor, get_device_count, get_dist_setting, get_logger, safe_snapshot_download
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_get_input_embeddings, patch_output_to_input_device
from ..register import ModelLoader, register_model

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


class ChatGLMLoader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        if model_kwargs.get('quantization_config') is not None:
            model_kwargs['quantization_config'].llm_int8_skip_modules = ['output_layer']
        model = super().get_model(model_dir, config, processor, model_kwargs)
        from torch.nn import CrossEntropyLoss
        __old_forward = CrossEntropyLoss.forward

        def cross_entropy_forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            target = target.to(device=inputs.device)
            return __old_forward(self, inputs, target)

        CrossEntropyLoss.forward = cross_entropy_forward
        return model

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        # fix transformers>=4.34 bug
        if version.parse(transformers.__version__) >= version.parse('4.34'):
            tokenizer_config = get_tokenizer_config(model_dir)
            class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
            tokenizer_cls: Type[PreTrainedTokenizerBase] = get_class_from_dynamic_module(class_ref, model_dir)
            tokenizer_cls._auto_class = 'AutoTokenizer'
            remove_property(tokenizer_cls, tokenizer_config)
            tokenizer = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
        else:
            tokenizer = super().get_processor(model_dir, config)
        _patch_tokenizer(tokenizer)
        return tokenizer


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
        ChatGLMLoader,
        template=TemplateType.chatglm2,
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
        ChatGLMLoader,
        template=TemplateType.chatglm4,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.chatglm))


class ChatGLM4Loader(ChatGLMLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer = super().get_processor(model_dir, config)
        if len(tokenizer.encode('<|user|>', add_special_tokens=False)) > 1:
            for k in tokenizer.special_tokens.keys():
                tokenizer.add_tokens(k)
        return tokenizer


register_model(
    ModelMeta(
        LLMModelType.chatglm4,
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
        ChatGLM4Loader,
        template=TemplateType.chatglm4,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.chatglm,
        requires=['transformers>=4.42'],
    ))

register_model(
    ModelMeta(
        LLMModelType.glm4,
        [
            ModelGroup([
                Model('ZhipuAI/GLM-4-9B-0414', 'zai-org/GLM-4-9B-0414'),
                Model('ZhipuAI/GLM-4-32B-0414', 'zai-org/GLM-4-32B-0414'),
                Model('ZhipuAI/GLM-4-32B-Base-0414', 'zai-org/GLM-4-32B-Base-0414'),
                Model('ZhipuAI/GLM-Z1-9B-0414', 'zai-org/GLM-Z1-9B-0414'),
                Model('ZhipuAI/GLM-Z1-32B-0414', 'zai-org/GLM-Z1-32B-0414'),
            ], TemplateType.glm4),
            ModelGroup([
                Model('ZhipuAI/GLM-Z1-Rumination-32B-0414', 'zai-org/GLM-Z1-Rumination-32B-0414'),
            ], TemplateType.glm4_z1_rumination)
        ],
        requires=['transformers>=4.51'],
        architectures=['Glm4ForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.codegeex4,
        [ModelGroup([
            Model('ZhipuAI/codegeex4-all-9b', 'zai-org/codegeex4-all-9b'),
        ])],
        ChatGLM4Loader,
        template=TemplateType.codegeex4,
        requires=['transformers<4.42'],
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.chatglm,
        tags=['coding'],
    ))


class ChatGLM4vLoader(ChatGLMLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
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
        return model

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        processor = super().get_processor(model_dir, config)
        processor.init_kwargs['image_size'] = 1120
        return processor


register_model(
    ModelMeta(
        MLLMModelType.chatglm4v,
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
        ChatGLM4vLoader,
        template=TemplateType.chatglm4v,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.chatglm4v,
    ))


class GLM4vLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Glm4vForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Glm4vForConditionalGeneration
        model = super().get_model(model_dir, *args, **kwargs)
        if hasattr(model, 'visual'):
            patch_get_input_embeddings(model.visual, 'patch_embed')
        return model


register_model(
    ModelMeta(
        MLLMModelType.glm4v,
        [
            ModelGroup(
                [
                    Model('ZhipuAI/GLM-4.1V-9B-Base', 'zai-org/GLM-4.1V-9B-Base'),
                    Model('ZhipuAI/GLM-4.1V-9B-Thinking', 'zai-org/GLM-4.1V-9B-Thinking'),
                    Model('ZhipuAI/AutoGLM-Phone-9B', 'zai-org/AutoGLM-Phone-9B')
                ],
                template=TemplateType.glm4v,
                requires=['transformers>=4.53'],
            ),
            ModelGroup(
                [
                    Model('ZhipuAI/Glyph', 'zai-org/Glyph'),
                ],
                template=TemplateType.glm4_5v,
                requires=['transformers>=4.57'],
            ),
            ModelGroup(
                [
                    Model('ZhipuAI/GLM-4.6V-Flash', 'zai-org/GLM-4.6V-Flash'),
                ],
                template=TemplateType.glm4_5v,
                requires=['transformers>=5.0.0.dev'],
            ),
        ],
        GLM4vLoader,
        model_arch=ModelArch.glm4v,
        architectures=['Glm4vForConditionalGeneration'],
    ))


class CogVLMLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        logger.warning('CogAgent with FusedLayerNorm will cause an training loss of NAN, '
                       'to avoid this, please uninstall apex.')
        logger.info('Please ignore the unimported warning.')
        return super().get_model(model_dir, *args, **kwargs)

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer_dir = safe_snapshot_download('AI-ModelScope/vicuna-7b-v1.5', download_model=False, check_local=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        return tokenizer


register_model(
    ModelMeta(
        MLLMModelType.cogvlm, [
            ModelGroup([
                Model('ZhipuAI/cogvlm-chat', 'zai-org/cogvlm-chat-hf'),
            ]),
        ],
        CogVLMLoader,
        template=TemplateType.cogvlm,
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
        CogVLMLoader,
        template=TemplateType.cogagent_chat,
        architectures=['CogAgentForCausalLM'],
        requires=['transformers<4.42', 'timm'],
        model_arch=ModelArch.cogvlm))

register_model(
    ModelMeta(
        MLLMModelType.cogagent_vqa, [ModelGroup([
            Model('ZhipuAI/cogagent-vqa', 'zai-org/cogagent-vqa-hf'),
        ])],
        CogVLMLoader,
        template=TemplateType.cogagent_vqa,
        architectures=['CogAgentForCausalLM'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.cogvlm))


class CogVLM2Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        # fix device map 4
        for layer in model.model.vision.transformer.layers:
            patch_output_to_input_device(layer.mlp)
            patch_output_to_input_device(layer.post_attention_layernorm)

        device = next(model.model.vision.linear_proj.parameters()).device
        model.model.vision.boi.data = model.model.vision.boi.to(device)
        model.model.vision.eoi.data = model.model.vision.eoi.to(device)
        return model


register_model(
    ModelMeta(
        MLLMModelType.cogvlm2, [
            ModelGroup([
                Model('ZhipuAI/cogvlm2-llama3-chat-19B', 'zai-org/cogvlm2-llama3-chat-19B'),
                Model('ZhipuAI/cogvlm2-llama3-chinese-chat-19B', 'zai-org/cogvlm2-llama3-chinese-chat-19B'),
            ]),
        ],
        CogVLM2Loader,
        template=TemplateType.cogvlm2,
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
        CogVLM2Loader,
        template=TemplateType.cogvlm2_video,
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
        template=TemplateType.chatglm4,
        architectures=['GlmForCausalLM'],
        requires=['transformers>=4.46'],
    ))


class GLMEdgeVLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        from transformers import AutoImageProcessor
        self.auto_tokenizer_cls = AutoImageProcessor
        return super().get_processor(model_dir, config)


register_model(
    ModelMeta(
        MLLMModelType.glm_edge_v,
        [
            ModelGroup([
                Model('ZhipuAI/glm-edge-v-2b', 'zai-org/glm-edge-v-2b'),
                Model('ZhipuAI/glm-edge-4b-chat', 'zai-org/glm-edge-4b-chat'),
            ]),
        ],
        GLMEdgeVLoader,
        template=TemplateType.glm_edge_v,
        architectures=['GlmForCausalLM'],
        requires=['transformers>=4.46'],
        model_arch=ModelArch.glm_edge_v,
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        LLMModelType.glm4_moe,
        [
            ModelGroup([
                Model('ZhipuAI/GLM-4.5-Air-Base', 'zai-org/GLM-4.5-Air-Base'),
                Model('ZhipuAI/GLM-4.5-Air', 'zai-org/GLM-4.5-Air'),
                Model('ZhipuAI/GLM-4.5-Air-FP8', 'zai-org/GLM-4.5-Air-FP8'),
                Model('ZhipuAI/GLM-4.5-Base', 'zai-org/GLM-4.5-Base'),
                Model('ZhipuAI/GLM-4.5', 'zai-org/GLM-4.5'),
                Model('ZhipuAI/GLM-4.5-FP8', 'zai-org/GLM-4.5-FP8'),
            ], TemplateType.glm4_5),
            ModelGroup([
                Model('ZhipuAI/GLM-4.6', 'zai-org/GLM-4.6'),
                Model('ZhipuAI/GLM-4.6-FP8', 'zai-org/GLM-4.6-FP8'),
            ], TemplateType.glm4_5),
            ModelGroup([
                Model('ZhipuAI/GLM-4.7', 'zai-org/GLM-4.7'),
                Model('ZhipuAI/GLM-4.7-FP8', 'zai-org/GLM-4.7-FP8'),
            ], TemplateType.glm4_7),
        ],
        requires=['transformers>=4.54'],
        architectures=['Glm4MoeForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.glm4_moe_lite,
        [
            ModelGroup([
                Model('ZhipuAI/GLM-4.7-Flash', 'zai-org/GLM-4.7-Flash'),
            ], TemplateType.glm4_7),
        ],
        requires=['transformers>=5.0.0.dev'],
        architectures=['Glm4MoeLiteForCausalLM'],
    ))


class Glm4vMoeLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Glm4vMoeForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Glm4vMoeForConditionalGeneration
        model = super().get_model(model_dir, *args, **kwargs)
        patch_get_input_embeddings(model.visual, 'patch_embed')
        return model


register_model(
    ModelMeta(
        MLLMModelType.glm4v_moe,
        [
            ModelGroup([
                Model('ZhipuAI/GLM-4.5V', 'zai-org/GLM-4.5V'),
                Model('ZhipuAI/GLM-4.5V-FP8', 'zai-org/GLM-4.5V-FP8'),
            ]),
            ModelGroup([
                Model('ZhipuAI/GLM-4.6V', 'zai-org/GLM-4.6V'),
                Model('ZhipuAI/GLM-4.6V-FP8', 'zai-org/GLM-4.6V-FP8'),
            ],
                       requires=['transformers>=5.0.0.dev']),
        ],
        Glm4vMoeLoader,
        template=TemplateType.glm4_5v,
        model_arch=ModelArch.glm4v,
        architectures=['Glm4vMoeForConditionalGeneration'],
        requires=['transformers>=4.56'],
    ))

# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType
from typing import Any, Dict

from transformers import PretrainedConfig, PreTrainedModel

from swift.template import TemplateType
from swift.utils import Processor, get_device, get_env_args
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_ignore_check_imports, patch_output_clone
from ..register import ModelLoader, register_model
from ..utils import use_submodel_func


class Phi3VisionLoader(ModelLoader):
    num_crops = 4

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        processor_kwargs = {'num_crops': get_env_args('num_crops', int, self.num_crops)}
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, **processor_kwargs)
        return processor

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_clone(model.model.vision_embed_tokens.wte)
        return model


register_model(
    ModelMeta(
        MLLMModelType.phi3_vision,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-vision-128k-instruct', 'microsoft/Phi-3-vision-128k-instruct'),
                Model('LLM-Research/Phi-3.5-vision-instruct', 'microsoft/Phi-3.5-vision-instruct'),
            ])
        ],
        Phi3VisionLoader,
        template=TemplateType.phi3_vision,
        architectures=['Phi3VForCausalLM'],
        model_arch=ModelArch.phi3_vision,
        requires=['transformers>=4.36'],
        tags=['vision'],
    ))


class Phi4MultimodalLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        processor = super().get_processor(model_dir, config)
        processor.audio_processor.audio_compression_rate = processor.audio_processor.compression_rate
        processor.audio_processor.audio_downsample_rate = processor.audio_processor.qformer_compression_rate
        processor.audio_processor.audio_feat_stride = processor.audio_processor.feat_stride
        del processor.audio_processor.feature_size
        del processor.audio_processor.sampling_rate
        del processor.audio_processor.padding_value
        del processor.__class__.chat_template
        processor.chat_template = None
        return processor

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        model.set_lora_adapter(['vision', 'speech'])
        return model


register_model(
    ModelMeta(
        MLLMModelType.phi4_multimodal,
        [ModelGroup([
            Model('LLM-Research/Phi-4-multimodal-instruct', 'microsoft/Phi-4-multimodal-instruct'),
        ])],
        Phi4MultimodalLoader,
        template=TemplateType.phi4_multimodal,
        architectures=['Phi4MMForCausalLM'],
        model_arch=ModelArch.phi4_multimodal,
        requires=['transformers>=4.36,<4.49', 'backoff', 'soundfile'],
        tags=['vision', 'audio'],
    ))


class FlorenceLoader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        config.vision_config.model_type = 'davit'  # fix merge-lora
        if model_kwargs['device_map'] == 'auto':
            model_kwargs['device_map'] = get_device()
        with patch_ignore_check_imports():
            model = super().get_model(model_dir, config, processor, model_kwargs)
        model.vision_tower.enable_checkpoint = True
        use_submodel_func(model, 'language_model', ['generate', 'forward'])
        return model


register_model(
    ModelMeta(
        MLLMModelType.florence,
        [
            # llama2
            ModelGroup([
                Model('AI-ModelScope/Florence-2-base-ft', 'microsoft/Florence-2-base-ft'),
                Model('AI-ModelScope/Florence-2-base', 'microsoft/Florence-2-base'),
                Model('AI-ModelScope/Florence-2-large', 'microsoft/Florence-2-large'),
                Model('AI-ModelScope/Florence-2-large-ft', 'microsoft/Florence-2-large-ft'),
            ]),
        ],
        FlorenceLoader,
        template=TemplateType.florence,
        architectures=['Florence2ForConditionalGeneration'],
        model_arch=ModelArch.florence,
        tags=['vision'],
    ))


class Phi3SmallLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)

        def rotary_emb(self, query_states, key_states, **kwargs):
            q_type = query_states.dtype
            k_type = key_states.dtype
            query_states, key_states = self.rotory_emb_origin(query_states, key_states, **kwargs)
            query_states = query_states.to(q_type)
            key_states = key_states.to(k_type)
            return query_states, key_states

        for i in range(32):  # TODO: 32
            re = model.model.layers[i].self_attn.rotary_emb
            re.rotory_emb_origin = re.forward
            re.forward = MethodType(rotary_emb, re)
        return model


register_model(
    ModelMeta(
        LLMModelType.phi3_small,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-small-8k-instruct', 'microsoft/Phi-3-small-8k-instruct'),
                Model('LLM-Research/Phi-3-small-128k-instruct', 'microsoft/Phi-3-small-128k-instruct'),
            ]),
        ],
        Phi3SmallLoader,
        template=TemplateType.phi3,
        architectures=['Phi3SmallForCausalLM'],
        model_arch=ModelArch.phi3_small,
        requires=['transformers>=4.36'],
    ))

register_model(
    ModelMeta(
        LLMModelType.phi2,
        [
            ModelGroup([
                Model('AI-ModelScope/phi-2', 'microsoft/phi-2'),
            ]),
        ],
        template=TemplateType.default,
        architectures=['PhiForCausalLM'],
        model_arch=ModelArch.phi2,
    ))

register_model(
    ModelMeta(
        LLMModelType.phi3,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct'),
                Model('LLM-Research/Phi-3-mini-128k-instruct', 'microsoft/Phi-3-mini-128k-instruct'),
                Model('LLM-Research/Phi-3-medium-4k-instruct', 'microsoft/Phi-3-medium-4k-instruct'),
                Model('LLM-Research/Phi-3-medium-128k-instruct', 'microsoft/Phi-3-medium-128k-instruct'),
                Model('LLM-Research/Phi-3.5-mini-instruct', 'microsoft/Phi-3.5-mini-instruct'),
            ]),
            ModelGroup([Model('LLM-Research/Phi-4-mini-instruct', 'microsoft/Phi-4-mini-instruct')])
        ],
        template=TemplateType.phi3,
        architectures=['Phi3ForCausalLM'],
        requires=['transformers>=4.36'],
        model_arch=ModelArch.phi3,
    ))

register_model(
    ModelMeta(
        LLMModelType.phi4,
        [
            ModelGroup([
                Model('LLM-Research/phi-4', 'microsoft/phi-4'),
            ]),
        ],
        template=TemplateType.phi4,
        architectures=['Phi3ForCausalLM'],
        requires=['transformers>=4.36'],
        model_arch=ModelArch.phi3,
    ))

register_model(
    ModelMeta(
        LLMModelType.phi3_moe,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3.5-MoE-instruct', 'microsoft/Phi-3.5-MoE-instruct'),
            ]),
        ],
        template=TemplateType.phi3,
        architectures=['PhiMoEForCausalLM'],
        requires=['transformers>=4.36'],
    ))

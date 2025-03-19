# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from types import MethodType
from typing import Any, Dict

from transformers import AutoConfig

from swift.llm import TemplateType
from swift.utils import get_device, get_env_args
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_ignore_check_imports, patch_output_clone
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, use_submodel_func


def get_model_tokenizer_phi3_vision(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    processor_kwargs = {}
    if 'num_crops' in kwargs:
        processor_kwargs['num_crops'] = get_env_args('num_crops', int, kwargs['num_crops'])
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, **processor_kwargs)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=processor.tokenizer, **kwargs)

    if load_model:
        patch_output_clone(model.model.vision_embed_tokens.wte)

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.phi3_vision,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-vision-128k-instruct', 'microsoft/Phi-3-vision-128k-instruct'),
                Model('LLM-Research/Phi-3.5-vision-instruct', 'microsoft/Phi-3.5-vision-instruct'),
            ])
        ],
        TemplateType.phi3_vision,
        partial(get_model_tokenizer_phi3_vision, num_crops=4),
        architectures=['Phi3VForCausalLM'],
        model_arch=ModelArch.phi3_vision,
        requires=['transformers>=4.36'],
        tags=['vision'],
    ))


def get_model_tokenizer_phi4_multimodal(*args, **kwargs):
    model, processor = get_model_tokenizer_multimodal(*args, **kwargs)
    processor.audio_processor.audio_compression_rate = processor.audio_processor.compression_rate
    processor.audio_processor.audio_downsample_rate = processor.audio_processor.qformer_compression_rate
    processor.audio_processor.audio_feat_stride = processor.audio_processor.feat_stride
    del processor.audio_processor.feature_size
    del processor.audio_processor.sampling_rate
    del processor.audio_processor.padding_value
    del processor.__class__.chat_template
    processor.chat_template = None
    if model is not None:
        model.set_lora_adapter(['vision', 'speech'])
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.phi4_multimodal,
        [ModelGroup([
            Model('LLM-Research/Phi-4-multimodal-instruct', 'microsoft/Phi-4-multimodal-instruct'),
        ])],
        TemplateType.phi4_multimodal,
        get_model_tokenizer_phi4_multimodal,
        architectures=['Phi4MMForCausalLM'],
        model_arch=ModelArch.phi4_multimodal,
        requires=['transformers>=4.36,<4.49', 'backoff', 'soundfile'],
        tags=['vision', 'audio'],
    ))


def get_model_tokenizer_florence(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.vision_config.model_type = 'davit'  # fix merge-lora
    if model_kwargs['device_map'] == 'auto':
        model_kwargs['device_map'] = get_device()
    kwargs['model_config'] = model_config
    with patch_ignore_check_imports():
        model, processor = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if model is not None:
        model.vision_tower.enable_checkpoint = True
        use_submodel_func(model, 'language_model', ['generate', 'forward'])
    return model, processor


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
        TemplateType.florence,
        get_model_tokenizer_florence,
        architectures=['Florence2ForConditionalGeneration'],
        model_arch=ModelArch.florence,
        tags=['vision'],
    ))


def get_model_tokenizer_phi3_small(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

    def rotary_emb(self, query_states, key_states, **kwargs):
        q_type = query_states.dtype
        k_type = key_states.dtype
        query_states, key_states = self.rotory_emb_origin(query_states, key_states, **kwargs)
        query_states = query_states.to(q_type)
        key_states = key_states.to(k_type)
        return query_states, key_states

    if model is not None:
        for i in range(32):
            re = model.model.layers[i].self_attn.rotary_emb
            re.rotory_emb_origin = re.forward
            re.forward = MethodType(rotary_emb, re)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.phi3_small,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-small-8k-instruct', 'microsoft/Phi-3-small-8k-instruct'),
                Model('LLM-Research/Phi-3-small-128k-instruct', 'microsoft/Phi-3-small-128k-instruct'),
            ]),
        ],
        TemplateType.phi3,
        get_model_tokenizer_phi3_small,
        architectures=['Phi3SmallForCausalLM'],
        model_arch=ModelArch.phi3_small,
        requires=['transformers>=4.36'],
    ))


def get_model_tokenizer_phi(model_dir: str,
                            model_info: ModelInfo,
                            model_kwargs: Dict[str, Any],
                            load_model: bool = True,
                            **kwargs):
    # TODO: check
    return get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.phi2,
        [
            ModelGroup([
                Model('AI-ModelScope/phi-2', 'microsoft/phi-2'),
            ]),
        ],
        TemplateType.default,
        get_model_tokenizer_phi,
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
            ModelGroup(Model('LLM-Research/Phi-4-mini-instruct', 'microsoft/Phi-4-mini-instruct'))
        ],
        TemplateType.phi3,
        get_model_tokenizer_with_flash_attn,
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
        TemplateType.phi4,
        get_model_tokenizer_with_flash_attn,
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
        TemplateType.phi3,
        get_model_tokenizer_with_flash_attn,
        architectures=['PhiMoEForCausalLM'],
        requires=['transformers>=4.36'],
    ))

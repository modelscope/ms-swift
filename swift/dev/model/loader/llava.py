# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the transformers-native LLaVA families (the ``*_hf`` checkpoints).

Ported from ``swift/model/models/llava.py`` -- only the checkpoints that load through a real
transformers ``Llava*ForConditionalGeneration`` class. They all share the ``llava_hf`` architecture
partition (``llava_next_video_hf`` is identical to it in transformers 5.x), so a single
``_LlavaHfBase`` carries ``model_arch``; each family declares its own ``model_cls`` / template /
version floor. Several families legitimately report the same ``architectures`` (six map to
``LlavaNextForConditionalGeneration``): that many-to-many reverse-lookup is intentional and matches
how llama/yi/codefuse already share ``LlamaForCausalLM`` -- id-match resolves them, and an unknown
checkpoint returns all candidates for the caller to disambiguate.

Not migrated here (bucket C -- see MODEL_MIGRATION.md): ``llama3_llava_next`` / ``llava1_6_mistral``
/ ``llava1_6_yi`` / ``llava_next_qwen`` (``LlavaLoader`` ``git_clone``s the haotian-liu/LLaVA-VL
repos, force-loads a CLIP vision tower, resizes embeddings and monkeypatches forward/generate) and
``llava_onevision1_5`` (dynamic-module class + ``_no_split_modules`` + ``patch_get_input_embeddings``).
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


class _LlavaHfBase(ModelLoader):
    """Shared ``llava_hf`` partition + multimodal flag; subclasses pin ``model_cls`` and template."""

    is_multimodal = True
    tags = ['vision']

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.llava_hf` / `.llava_next_video_hf` (identical), transformers>=4.52 branch.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )


@register_model
class LlavaLlama3HfLoader(_LlavaHfBase):
    model_type = 'llava_llama3_hf'
    model_cls = 'transformers:LlavaForConditionalGeneration'
    # This checkpoint's config does not auto-resolve to LlavaConfig, so it is pinned (legacy set it).
    config_cls = 'transformers:LlavaConfig'
    architectures = ['LlavaForConditionalGeneration']
    template = 'llava_llama3_hf'
    requires = ['transformers>=4.36']
    models = [('AI-ModelScope/llava-llama-3-8b-v1_1-transformers', 'xtuner/llava-llama-3-8b-v1_1-transformers')]


@register_model
class Llava1_5HfLoader(_LlavaHfBase):
    model_type = 'llava1_5_hf'
    model_cls = 'transformers:LlavaForConditionalGeneration'
    architectures = ['LlavaForConditionalGeneration']
    template = 'llava1_5_hf'
    requires = ['transformers>=4.36']
    models = ['llava-hf/llava-1.5-7b-hf', 'llava-hf/llava-1.5-13b-hf']


@register_model
class LlavaOnevisionHfLoader(_LlavaHfBase):
    model_type = 'llava_onevision_hf'
    model_cls = 'transformers:LlavaOnevisionForConditionalGeneration'
    architectures = ['LlavaOnevisionForConditionalGeneration']
    template = 'llava_onevision_hf'
    requires = ['transformers>=4.45']
    tags = ['vision', 'video']
    models = [
        'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
        'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        'llava-hf/llava-onevision-qwen2-72b-ov-hf',
    ]


class _LlavaNextHfLoader(_LlavaHfBase):
    model_cls = 'transformers:LlavaNextForConditionalGeneration'
    architectures = ['LlavaNextForConditionalGeneration']
    requires = ['transformers>=4.39']


@register_model
class LlavaNextQwenHfLoader(_LlavaNextHfLoader):
    model_type = 'llava_next_qwen_hf'
    template = 'llava_next_qwen_hf'
    models = ['llava-hf/llava-next-72b-hf', 'llava-hf/llava-next-110b-hf']


@register_model
class Llama3LlavaNextHfLoader(_LlavaNextHfLoader):
    model_type = 'llama3_llava_next_hf'
    template = 'llama3_llava_next_hf'
    models = ['llava-hf/llama3-llava-next-8b-hf']


@register_model
class Llava1_6VicunaHfLoader(_LlavaNextHfLoader):
    model_type = 'llava1_6_vicuna_hf'
    template = 'llava1_6_vicuna_hf'
    models = ['llava-hf/llava-v1.6-vicuna-7b-hf', 'llava-hf/llava-v1.6-vicuna-13b-hf']


@register_model
class Llava1_6MistralHfLoader(_LlavaNextHfLoader):
    model_type = 'llava1_6_mistral_hf'
    template = 'llava1_6_mistral_hf'
    models = ['llava-hf/llava-v1.6-mistral-7b-hf']


@register_model
class LlavaLlama3_1HfLoader(_LlavaNextHfLoader):
    model_type = 'llava_llama3_1_hf'
    template = 'llava_llama3_1_hf'
    requires = ['transformers>=4.41']
    models = ['swift/llava-llama3.1-8b']


@register_model
class Llava1_6YiHfLoader(_LlavaNextHfLoader):
    # legacy defines a LlavaNextYiHfLoader that pins image_token_index=64003, but the registration
    # wires the plain LlavaNextHfLoader -- that dead-code override never ran, so it is not ported.
    model_type = 'llava1_6_yi_hf'
    template = 'llava1_6_yi_hf'
    models = ['llava-hf/llava-v1.6-34b-hf']


@register_model
class LlavaNextVideoHfLoader(_LlavaHfBase):
    model_type = 'llava_next_video_hf'
    model_cls = 'transformers:LlavaNextVideoForConditionalGeneration'
    architectures = ['LlavaNextVideoForConditionalGeneration']
    template = 'llava_next_video_hf'
    requires = ['transformers>=4.42', 'av']
    tags = ['video']
    models = [
        'llava-hf/LLaVA-NeXT-Video-7B-DPO-hf',
        'llava-hf/LLaVA-NeXT-Video-7B-32K-hf',
        'llava-hf/LLaVA-NeXT-Video-7B-hf',
    ]


@register_model
class LlavaNextVideoYiHfLoader(LlavaNextVideoHfLoader):
    """The 34B Yi checkpoint shifts its special-token ids; it shares the ``llava_next_video_hf``
    template but must override the token indices in config."""

    model_type = 'llava_next_video_yi_hf'
    architectures = []
    models = ['llava-hf/LLaVA-NeXT-Video-34B-hf']

    def process_config(self, config):
        config.video_token_index = 64003
        config.image_token_index = 64004
        return config

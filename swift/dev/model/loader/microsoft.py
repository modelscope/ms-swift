# Copyright (c) ModelScope Contributors. All rights reserved.
# Phi text families + phi3_vision + phi4_multimodal (phi3_small dropped, MODEL_MIGRATION.md).
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class Phi2Loader(ModelLoader):

    model_type = 'phi2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['PhiForCausalLM']
    template = 'default'
    models = [('AI-ModelScope/phi-2', 'microsoft/phi-2')]


@register_model
class Phi3Loader(ModelLoader):

    model_type = 'phi3'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['Phi3ForCausalLM']
    template = 'phi3'
    requires = ['transformers>=4.36']
    models = [
        ('LLM-Research/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct'),
        ('LLM-Research/Phi-3-mini-128k-instruct', 'microsoft/Phi-3-mini-128k-instruct'),
        ('LLM-Research/Phi-3-medium-4k-instruct', 'microsoft/Phi-3-medium-4k-instruct'),
        ('LLM-Research/Phi-3-medium-128k-instruct', 'microsoft/Phi-3-medium-128k-instruct'),
        ('LLM-Research/Phi-3.5-mini-instruct', 'microsoft/Phi-3.5-mini-instruct'),
        ('LLM-Research/Phi-4-mini-instruct', 'microsoft/Phi-4-mini-instruct'),
    ]


@register_model
class Phi4Loader(Phi3Loader):
    """phi-4 shares ``Phi3ForCausalLM`` with phi3; only the chat template differs."""

    model_type = 'phi4'
    architectures = []
    template = 'phi4'
    models = [('LLM-Research/phi-4', 'microsoft/phi-4')]


@register_model
class Phi3MoeLoader(ModelLoader):

    model_type = 'phi3_moe'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['PhiMoEForCausalLM']
    template = 'phi3'
    requires = ['transformers>=4.36']
    models = [('LLM-Research/Phi-3.5-MoE-instruct', 'microsoft/Phi-3.5-MoE-instruct')]


@register_model
class Phi3VisionLoader(ModelLoader):
    """Phi-3-vision: remote-code ``Phi3VForCausalLM``. Legacy also called
    ``patch_output_clone(vision_embed_tokens.wte)`` -- obsolete per PATCH_INVENTORY, so dropped."""

    model_type = 'phi3_vision'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['Phi3VForCausalLM']
    template = 'phi3_vision'
    requires = ['transformers>=4.36']
    tags = ['vision']
    is_multimodal = True
    #: number of image crops; a user knob read from the ``num_crops`` env var (legacy default 4).
    num_crops = 4
    models = [
        ('LLM-Research/Phi-3-vision-128k-instruct', 'microsoft/Phi-3-vision-128k-instruct'),
        ('LLM-Research/Phi-3.5-vision-instruct', 'microsoft/Phi-3.5-vision-instruct'),
    ]

    def build_processor(self, model_dir, config, **kwargs):
        from swift.dev.utils import get_env_args
        # legacy passed num_crops to AutoProcessor; keep the env knob, let trust_remote_code (base)
        # and the preprocessor-config detection pick AutoProcessor.
        kwargs.setdefault('num_crops', get_env_args('num_crops', int, self.num_crops))
        return super().build_processor(model_dir, config, **kwargs)

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model='model.layers',
            aligner='model.vision_embed_tokens.img_projection',
            vision_tower='model.vision_embed_tokens.img_processor',
        )


@register_model
class Phi4MultimodalLoader(ModelLoader):
    """Phi-4-multimodal via **transformers' own** class, not the checkpoint's remote code.

    Re-basing drops the transformers<4.49 ceiling, the six audio_processor rewrites (in-tree uses
    settled names) and set_lora_adapter (no LoRA machinery here; modality tuning becomes ordinary
    full/LoRA tuning over ``embed_tokens_extend``). Unverified: whether the checkpoint's
    ``config.architectures`` reads the new name; if not, ``--model_type``.
    """

    model_type = 'phi4_multimodal'
    model_cls = 'transformers:Phi4MultimodalForCausalLM'
    architectures = ['Phi4MultimodalForCausalLM']
    template = 'phi4_multimodal'
    requires = ['transformers>=5.0', 'backoff', 'soundfile']
    tags = ['vision', 'audio']
    is_multimodal = True
    models = [('LLM-Research/Phi-4-multimodal-instruct', 'microsoft/Phi-4-multimodal-instruct')]

    @property
    def model_arch(self) -> ModelArch:
        embed = 'model.embed_tokens_extend'
        return ModelArch(
            language_model=['model.layers', 'lm_head'],
            aligner=[
                f'{embed}.image_embed.img_projection_up',
                f'{embed}.image_embed.img_projection_down',
                f'{embed}.audio_embed.up_proj_for_speech',
                f'{embed}.audio_embed.down_proj_for_speech',
            ],
            vision_tower=[f'{embed}.image_embed.img_processor', f'{embed}.audio_embed.encoder'],
        )

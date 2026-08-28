# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Microsoft Phi text families.

Ported from ``swift/model/models/microsoft.py`` -- the plain causal-LM Phi checkpoints plus the
remote-code ``phi3_vision`` (``Phi3VForCausalLM`` via ``AutoModelForCausalLM`` + ``trust_remote_code``).

Not migrated here (see MODEL_MIGRATION.md): ``florence`` (moved to ``mllm.py``) and ``phi3_small``
(a per-layer ``rotary_emb.forward`` dtype patch over a hardcoded 32 layers).

``phi4_multimodal`` *is* migrated (bottom of this file), but re-based onto transformers' own
implementation rather than the checkpoint's remote code -- see that class for what changes.
"""
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
    """Phi-4-multimodal, loaded through **transformers' own** ``Phi4MultimodalForCausalLM`` rather than
    the checkpoint's remote code. That re-basing is the whole point of this loader and it changes three
    things relative to legacy:

    * **Version.** Legacy pinned ``transformers>=4.36,<4.49`` because it drove the remote-code
      ``Phi4MMForCausalLM``. transformers ships its own port (``transformers.models.phi4_multimodal``),
      so the ceiling disappears and no ``trust_remote_code`` is needed.
    * **Processor.** Legacy rewrote six attributes on ``processor.audio_processor``
      (``compression_rate`` -> ``audio_compression_rate`` etc.), deleted three more, and nulled the
      class-level ``chat_template``. The in-tree ``Phi4MultimodalProcessor`` already uses the settled
      field names, so all of that is dropped.
    * **Modality LoRA is gone.** Legacy called ``model.set_lora_adapter(['vision', 'speech'])``: the
      remote-code model carried *built-in* vision/speech LoRA adapters that had to be activated.
      transformers' port has no LoRA machinery at all (verified: ``set_lora_adapter`` is absent and the
      module source contains zero ``lora`` references) -- the modality weights are plain dense layers.
      **Fine-tuning semantics therefore differ from legacy**: what used to be adapter activation is now
      ordinary full/LoRA tuning over ``embed_tokens_extend``, chosen by the usual tuner arguments.

    ``model_arch`` follows the in-tree layout, which also renamed the projections: the single
    ``img_projection`` / ``audio_projection`` became up/down pairs.

    Unverified: whether the published checkpoint's ``config.architectures`` already reads
    ``Phi4MultimodalForCausalLM`` (post-port) or still the remote-code ``Phi4MMForCausalLM``. If a
    download resolves to the old name, reverse-lookup will miss and ``--model_type phi4_multimodal``
    is needed explicitly.
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

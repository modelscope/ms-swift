# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the MiniCPM-V / MiniCPM-o multimodal families (from ``swift/model/models/minicpm.py``).

These are outer-wrapper remote-code checkpoints: the real LLM is ``model.llm``, so top-level calls
are delegated there (``delegate_to_submodel``). Unlike Ovis, MiniCPM-V also binds two of the *model's*
methods onto the *processor* (``get_slice_image_placeholder`` / ``transform``); that coupling needs
both objects at once, so it lives in ``build_model`` (which receives ``processor``) rather than
``process_model``. Legacy's ``_patch_minicpmv_device_map`` and ``patch_output_clone`` are obsolete per
PATCH_INVENTORY and dropped.

Not migrated here (see MODEL_MIGRATION.md):
  * ``minicpmo``'s 4_5 group -- pinned ``transformers==4.51.3`` + extra ``minicpmo-utils`` dep.
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class MiniCPMVLoader(ModelLoader):
    """MiniCPM-V (v1/v2): remote-code ``MiniCPMV``. The resampler is force-aligned to the model dtype
    (legacy 'fix float32'), the inner ``llm`` handles the real language calls, and the model's slicing
    helpers are bound onto the processor when present."""

    model_type = 'minicpmv'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['MiniCPMV']
    template = 'minicpmv'
    requires = ['timm', 'transformers<4.42']
    tags = ['vision']
    is_multimodal = True
    models = [('OpenBMB/MiniCPM-V', 'openbmb/MiniCPM-V'), ('OpenBMB/MiniCPM-V-2', 'openbmb/MiniCPM-V-2')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model='llm', aligner='resampler', vision_tower='vpm')

    def build_model(self, model_dir, config, processor, **kwargs):
        model = super().build_model(model_dir, config, processor, **kwargs)
        model.resampler.to(model.dtype)  # legacy 'fix float32'
        self.delegate_to_submodel(model, 'llm', ['generate', 'forward', 'get_input_embeddings'])
        if hasattr(model, 'get_slice_image_placeholder'):
            from types import MethodType
            # MiniCPM-V's image slicing lives on the model; the template reaches it through the
            # processor, so bind the two methods across (needs both objects -> here, not process_model).
            processor.get_slice_image_placeholder = MethodType(model.get_slice_image_placeholder, processor)
            processor.transform = MethodType(model.transform, processor)
        return model


@register_model
class MiniCPMV2_5Loader(MiniCPMVLoader):
    model_type = 'minicpmv2_5'
    template = 'minicpmv2_5'
    requires = ['timm', 'transformers>=4.36']
    models = [('OpenBMB/MiniCPM-Llama3-V-2_5', 'openbmb/MiniCPM-Llama3-V-2_5')]


@register_model
class MiniCPMV2_6Loader(MiniCPMVLoader):
    model_type = 'minicpmv2_6'
    template = 'minicpmv2_6'
    requires = ['timm', 'transformers>=4.36', 'decord']
    tags = ['vision', 'video']
    models = [('OpenBMB/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-2_6')]


@register_model
class MiniCPMV4Loader(MiniCPMVLoader):
    model_type = 'minicpmv4'
    template = 'minicpmv4'
    requires = ['timm', 'transformers>=4.36', 'decord']
    tags = ['vision', 'video']
    models = [('OpenBMB/MiniCPM-V-4', 'openbmb/MiniCPM-V-4')]


@register_model
class MiniCPMV4_5Loader(MiniCPMVLoader):
    model_type = 'minicpmv4_5'
    template = 'minicpmv4_5'
    requires = ['timm', 'transformers>=4.36', 'decord']
    tags = ['vision', 'video']
    models = [('OpenBMB/MiniCPM-V-4_5', 'openbmb/MiniCPM-V-4_5')]


@register_model
class MiniCPMV4_6Loader(ModelLoader):
    """MiniCPM-V-4.6: unlike the earlier remote-code ``MiniCPMV`` wrappers, this one is transformers-
    native (``MiniCPMV4_6ForConditionalGeneration``, needs >=5.7.0) and loads through
    ``AutoModelForImageTextToText`` with the standard ``model.language_model`` layout -- so no
    ``delegate_to_submodel`` and no processor-method binding.

    Its LLM uses qwen3.5-style linear attention, whose sequence-parallel path needs a live global
    patch (``_patch_qwen3_5_linear_attention_sequence_parallel``); this is a real runtime requirement,
    not an obsolete device_map patch, so it is applied in ``build_model`` exactly as the qwen3.5
    loaders do.
    """

    model_type = 'minicpmv4_6'
    model_cls = 'transformers:AutoModelForImageTextToText'
    architectures = ['MiniCPMV4_6ForConditionalGeneration']
    template = 'minicpmv4_6'
    requires = ['transformers>=5.7.0']
    tags = ['vision', 'video']
    is_multimodal = True
    models = [('OpenBMB/MiniCPM-V-4.6', 'openbmb/MiniCPM-V-4.6')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model'],
            aligner=['model.merger'],
            vision_tower=['model.vision_tower'],
        )

    def build_model(self, model_dir, config, processor, **kwargs):
        from swift.model.models.qwen import _patch_qwen3_5_linear_attention_sequence_parallel
        _patch_qwen3_5_linear_attention_sequence_parallel()
        return super().build_model(model_dir, config, processor, **kwargs)


@register_model
class MiniCPMOLoader(MiniCPMVLoader):
    """MiniCPM-o (omni): adds an audio tower and gates TTS/audio init via env vars on the config."""

    model_type = 'minicpmo'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MiniCPMO']
    template = 'minicpmo'
    requires = ['timm', 'transformers>=4.36', 'decord', 'soundfile']
    tags = ['vision', 'video', 'omni', 'audio']
    models = [('OpenBMB/MiniCPM-o-2_6', 'openbmb/MiniCPM-o-2_6')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model='llm', aligner='resampler', vision_tower=['vpm', 'apm'])

    def process_config(self, config):
        from swift.dev.utils import get_env_args
        from transformers.utils import strtobool
        config.init_tts = strtobool(get_env_args('init_tts', str, 'false'))
        config.init_audio = strtobool(get_env_args('init_audio', str, 'true'))
        return config


@register_model
class MiniCPMO4_5Loader(MiniCPMOLoader):
    """MiniCPM-o-4_5: same remote-code ``MiniCPMO`` class and module layout as 2_6, but its own chat
    template -- hence a template variant with ``architectures=[]`` (2_6 keeps ownership of the class
    name for reverse-lookup).

    **This checkpoint is not usable on transformers 5.5.** Legacy pins the group at
    ``transformers==4.51.3`` and additionally requires ``minicpmo-utils==1.0.6`` (not installed here).
    Registering it anyway is deliberate: ``--model_type minicpmo4_5`` and id matching resolve, and the
    version check then reports the real conflict -- exactly the legacy behaviour. The pin is left as
    legacy wrote it rather than optimistically widened, because unlike ``qwen3_asr`` there is no
    evidence about what the pin actually guards.
    """

    model_type = 'minicpmo4_5'
    architectures = []
    template = 'minicpmo4_5'
    requires = ['timm', 'transformers==4.51.3', 'decord', 'soundfile', 'minicpmo-utils==1.0.6']
    models = [('OpenBMB/MiniCPM-o-4_5', 'openbmb/MiniCPM-o-4_5')]


# ============================ MiniCPM text families ============================


@register_model
class MiniCPMTextLoader(ModelLoader):
    """MiniCPM dense text (2B/1B); reverse-lookup owner for ``MiniCPMForCausalLM``."""

    model_type = 'minicpm'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['MiniCPMForCausalLM']
    template = 'minicpm'
    requires = ['transformers>=4.36.0']
    models = [
        ('OpenBMB/MiniCPM-2B-sft-fp32', 'openbmb/MiniCPM-2B-sft-fp32'),
        ('OpenBMB/MiniCPM-2B-dpo-fp32', 'openbmb/MiniCPM-2B-dpo-fp32'),
        ('OpenBMB/MiniCPM-1B-sft-bf16', 'openbmb/MiniCPM-1B-sft-bf16'),
    ]


@register_model
class MiniCPMChatMLLoader(MiniCPMTextLoader):
    model_type = 'minicpm_chatml'
    architectures = ['MiniCPMForCausalLM']
    template = 'chatml'
    requires = ['transformers>=4.36']
    models = [
        ('OpenBMB/MiniCPM-2B-128k', 'openbmb/MiniCPM-2B-128k'),
        ('OpenBMB/MiniCPM4-0.5B', 'openbmb/MiniCPM4-0.5B'),
        ('OpenBMB/MiniCPM4-8B', 'openbmb/MiniCPM4-8B'),
    ]


@register_model
class MiniCPMMoeLoader(MiniCPMTextLoader):
    model_type = 'minicpm_moe'
    architectures = ['MiniCPMForCausalLM']
    template = 'minicpm'
    requires = ['transformers>=4.36']
    is_moe = True
    models = [('OpenBMB/MiniCPM-MoE-8x2B', 'openbmb/MiniCPM-MoE-8x2B')]


@register_model
class MiniCPM3Loader(ModelLoader):
    model_type = 'minicpm3'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['MiniCPM3ForCausalLM']
    template = 'chatml'
    requires = ['transformers>=4.36']
    models = [('OpenBMB/MiniCPM3-4B', 'openbmb/MiniCPM3-4B')]


@register_model
class MiniCPM5Loader(ModelLoader):
    """MiniCPM5-1B. Unlike its older siblings this is a plain in-tree ``LlamaForCausalLM`` checkpoint --
    legacy registered it as a ``minicpm5``-template group *under* ``llama``, not under the MiniCPM
    model_type. It lands here (with the family it belongs to by name) as an ``architectures=[]``
    template variant, so reverse-lookup for ``LlamaForCausalLM`` still resolves to ``llama``.
    """

    model_type = 'minicpm5'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = []  # template variant of the llama architecture; owner stays `llama`
    template = 'minicpm5'
    requires = ['transformers>=5.6']
    models = [
        ('OpenBMB/MiniCPM5-1B', 'openbmb/MiniCPM5-1B'),
        ('OpenBMB/MiniCPM5-1B-Base', 'openbmb/MiniCPM5-1B-Base'),
        ('OpenBMB/MiniCPM5-1B-SFT', 'openbmb/MiniCPM5-1B-SFT'),
    ]

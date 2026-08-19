# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Qwen families that need more than the plain transformers path.

Ported from ``swift/model/models/qwen.py`` against the ``ModelLoader`` seams in ``.base``.
Every ``PORT`` comment marks a place where the current seams did not fit; they are collected
in one list at the bottom of this file rather than being silently worked around.

A *template variant* -- a checkpoint that loads exactly like its parent but speaks a different chat
format (QVQ, MiMo-VL, gme, OvisOCR2) -- is its own thin subclass, not a nested ``ModelGroup``. It
declares ``model_type`` + ``template`` + ``models`` and, crucially, leaves ``architectures`` empty:
reverse-lookup from a HuggingFace class name must land on the one base family, never on a variant,
so variants are reachable only by id match or an explicit ``--model_type``.
"""
from __future__ import annotations

import importlib.metadata
import os
from typing import Dict

from packaging import version
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.versions import require_version

from swift.utils import get_logger
from ..keep_alive import apply_vision_keep_alive
from .base import ModelArch, ModelLoader, register_model

logger = get_logger()


@register_model
class Qwen2VLLoader(ModelLoader):
    """Qwen2-VL: the root of the whole Qwen vision chain."""

    model_type = 'qwen2_vl'
    model_cls = 'transformers:Qwen2VLForConditionalGeneration'
    architectures = ['Qwen2VLForConditionalGeneration']
    template = 'qwen2_vl'
    requires = ['transformers>=4.45', 'qwen_vl_utils>=0.0.6', 'decord']
    tags = ['vision', 'video']
    is_multimodal = True
    models = [
        'Qwen/Qwen2-VL-2B-Instruct',
        'Qwen/Qwen2-VL-7B-Instruct',
        'Qwen/Qwen2-VL-72B-Instruct',
        'Qwen/Qwen2-VL-2B',
        'Qwen/Qwen2-VL-7B',
        'Qwen/Qwen2-VL-72B',
        'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4',
        'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4',
        'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4',
        'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8',
        'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8',
        'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8',
        'Qwen/Qwen2-VL-2B-Instruct-AWQ',
        'Qwen/Qwen2-VL-7B-Instruct-AWQ',
        'Qwen/Qwen2-VL-72B-Instruct-AWQ',
        'bytedance-research/UI-TARS-2B-SFT',
        'bytedance-research/UI-TARS-7B-SFT',
        'bytedance-research/UI-TARS-7B-DPO',
        'bytedance-research/UI-TARS-72B-SFT',
        'bytedance-research/UI-TARS-72B-DPO',
        'allenai/olmOCR-7B-0225-preview',
        ('OpenDataLab/MinerU2.5-Pro-2604-1.2B', 'opendatalab/MinerU2.5-Pro-2604-1.2B'),
    ]

    # PORT-3: `is_moe` was `getattr(self, 'is_moe', False)` in legacy -- the author wanted a
    # declaration but had nowhere to put it, so it was read defensively off the instance.
    is_moe = False
    # PORT-4: the patch_size a *third-party library* must be configured with. Not config, not
    # processor, not model -- there is no seam for "configure the outside world".
    image_patch_size = 14
    # PORT-5: env defaults that only exist to steer qwen_vl_utils. Qwen3VLEmb/Reranker set three.
    env_defaults: Dict[str, str] = {}

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            vision_tower='model.visual',
            aligner='model.visual.merger',
        )

    def setup_environment(self) -> None:
        """PORT-6: a fourth seam the design is missing.

        Legacy called this ``_check_qwen_vl_utils`` and four classes in the chain override it.
        It must run *before* the processor is built, does version gating, sets env defaults and
        reaches into ``qwen_vl_utils.vision_process`` module globals. It is neither a ``build``
        nor a ``process`` of config/tokenizer/model.
        """
        for key, value in self.env_defaults.items():
            os.environ.setdefault(key, value)
        try:
            utils_version = importlib.metadata.version('qwen_vl_utils')
        except importlib.metadata.PackageNotFoundError:
            raise importlib.metadata.PackageNotFoundError(
                "The 'qwen_vl_utils' distribution was not found and is required by this application.")
        if version.parse(utils_version) >= version.parse('0.0.14'):
            self._compat_qwen_vl_utils(self.image_patch_size)
        else:
            require_version('qwen_vl_utils<0.0.12')

    @staticmethod
    def _compat_qwen_vl_utils(image_patch_size: int) -> None:
        """Translate the pixel-count env vars into the token-count ones qwen_vl_utils>=0.0.14 reads.

        The library switched its knobs from pixels to visual tokens; users (and every example
        script) still export ``MAX_PIXELS``. One visual token covers
        ``(patch_size * spatial_merge_size)**2`` pixels, so the conversion is a plain divide. An
        already-set token var always wins -- the pixel var is only a fallback.
        """
        spatial_merge_size = int(os.getenv('SPATIAL_MERGE_SIZE', '2'))
        image_factor = image_patch_size * spatial_merge_size
        pixels_to_tokens = {
            'MAX_PIXELS': 'IMAGE_MAX_TOKEN_NUM',
            'MIN_PIXELS': 'IMAGE_MIN_TOKEN_NUM',
            'VIDEO_MAX_PIXELS': 'VIDEO_MAX_TOKEN_NUM',
            'VIDEO_MIN_PIXELS': 'VIDEO_MIN_TOKEN_NUM',
        }
        for source_var, target_var in pixels_to_tokens.items():
            value = os.getenv(source_var)
            if value and not os.getenv(target_var):
                os.environ[target_var] = str(int(value) // image_factor**2)

    def build_processor(self, model_dir: str, config: PretrainedConfig, **kwargs):
        # PORT-6 fallout: with no dedicated seam, the environment setup has to be smuggled into
        # the front of build_processor, where a subclass that overrides build_processor without
        # calling super() silently loses it -- exactly the failure mode we set out to remove.
        self.setup_environment()
        return super().build_processor(model_dir, config, **kwargs)

    def process_tokenizer(self, processor):
        from swift.model.models.qwen import patch_qwen_vl_utils
        from qwen_vl_utils import vision_process
        # In order to have different hashes for the template.
        processor.global_vars = patch_qwen_vl_utils(vision_process)
        return processor

    def process_model(self, model):
        # DeepSpeed ZeRO-3 keep-alive, driven entirely by `ModelArch.aligner` -- no forward fork.
        # On a text-only micro-batch the stock forward would skip the vision tower, so its params go
        # dark on that rank and the cross-rank all-gather / reduce-scatter deadlocks. The data side
        # feeds a minimal dummy image so the *stock* forward runs the whole vision path; these hooks
        # zero the aligner output so the dummy contributes exactly nothing. Because the aligner is
        # the single chokepoint between vision tower and LLM, this keeps the tower alive while
        # blocking all contamination -- and it reads the aligner names off `model_arch`, so every VL
        # subclass gets the right modules automatically (Qwen3-VL's deepstack mergers included,
        # Qwen3.5's qwen2_vl-style single merger too). Replaces legacy `_compat_qwen3_vl_mixed_data`,
        # a verbatim forward fork pinned to a dozen transformers internals.
        model._vision_keep_alive = apply_vision_keep_alive(model, self.model_arch.aligner)
        return model


@register_model
class QVQLoader(Qwen2VLLoader):
    """QVQ-72B: Qwen2-VL loading, a reasoning-specific chat template."""

    model_type = 'qvq'
    template = 'qvq'
    architectures = []  # reachable by id / explicit --model_type only; base owns the reverse-lookup
    models = ['Qwen/QVQ-72B-Preview']


@register_model
class Qwen2GmeLoader(Qwen2VLLoader):
    """gme embedding checkpoints: Qwen2-VL loading, the gme template."""

    model_type = 'qwen2_gme'
    template = 'qwen2_gme'
    architectures = []
    tags = ['vision']
    models = [
        ('iic/gme-Qwen2-VL-2B-Instruct', 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct'),
        ('iic/gme-Qwen2-VL-7B-Instruct', 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct'),
    ]


@register_model
class Qwen2_5VLLoader(Qwen2VLLoader):
    """Only the model class differs from Qwen2-VL -- one declaration, no method left."""

    model_type = 'qwen2_5_vl'
    model_cls = 'transformers:Qwen2_5_VLForConditionalGeneration'
    architectures = ['Qwen2_5_VLForConditionalGeneration']
    template = 'qwen2_5_vl'
    requires = ['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord']
    models = [
        'Qwen/Qwen2.5-VL-3B-Instruct',
        'Qwen/Qwen2.5-VL-7B-Instruct',
        'Qwen/Qwen2.5-VL-32B-Instruct',
        'Qwen/Qwen2.5-VL-72B-Instruct',
        'Qwen/Qwen2.5-VL-3B-Instruct-AWQ',
        'Qwen/Qwen2.5-VL-7B-Instruct-AWQ',
        'Qwen/Qwen2.5-VL-32B-Instruct-AWQ',
        'Qwen/Qwen2.5-VL-72B-Instruct-AWQ',
    ]


@register_model
class MiMoVLLoader(Qwen2_5VLLoader):
    """Xiaomi MiMo-VL: Qwen2.5-VL loading, its own chat template."""

    model_type = 'mimo_vl'
    template = 'mimo_vl'
    architectures = []
    models = ['XiaomiMiMo/MiMo-VL-7B-SFT', 'XiaomiMiMo/MiMo-VL-7B-RL']


@register_model
class Qwen3VLLoader(Qwen2VLLoader):

    model_type = 'qwen3_vl'
    model_cls = 'transformers:Qwen3VLForConditionalGeneration'
    architectures = ['Qwen3VLForConditionalGeneration']
    template = 'qwen3_vl'
    requires = ['transformers>=4.57', 'qwen_vl_utils>=0.0.14', 'decord']
    image_patch_size = 16
    # PORT-7: Qwen3-VL requires qwen_vl_utils>=0.0.14 unconditionally, while Qwen2-VL tolerates
    # <0.0.12. A declared `requires` cannot express "and therefore skip the version fork", so
    # the fork stays imperative below.
    min_qwen_vl_utils = '0.0.14'
    models = [
        'Qwen/Qwen3-VL-2B-Instruct',
        'Qwen/Qwen3-VL-2B-Thinking',
        'Qwen/Qwen3-VL-2B-Instruct-FP8',
        'Qwen/Qwen3-VL-2B-Thinking-FP8',
        'Qwen/Qwen3-VL-4B-Instruct',
        'Qwen/Qwen3-VL-4B-Thinking',
        'Qwen/Qwen3-VL-4B-Instruct-FP8',
        'Qwen/Qwen3-VL-4B-Thinking-FP8',
        'Qwen/Qwen3-VL-8B-Instruct',
        'Qwen/Qwen3-VL-8B-Thinking',
        'Qwen/Qwen3-VL-8B-Instruct-FP8',
        'Qwen/Qwen3-VL-8B-Thinking-FP8',
        'Qwen/Qwen3-VL-32B-Instruct',
        'Qwen/Qwen3-VL-32B-Thinking',
        'Qwen/Qwen3-VL-32B-Instruct-FP8',
        'Qwen/Qwen3-VL-32B-Thinking-FP8',
    ]

    def setup_environment(self) -> None:
        for key, value in self.env_defaults.items():
            os.environ.setdefault(key, value)
        require_version(f'qwen_vl_utils>={self.min_qwen_vl_utils}')
        self._compat_qwen_vl_utils(self.image_patch_size)

    @property
    def model_arch(self) -> ModelArch:
        # Qwen3-VL adds the deepstack mergers, so the aligner is no longer a single prefix. The base
        # `process_model` reads this and hooks all of them for the keep-alive, so no override here.
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            vision_tower='model.visual',
            aligner=['model.visual.merger', 'model.visual.deepstack_merger_list'],
        )


@register_model
class Qwen3VLMoeLoader(Qwen3VLLoader):

    model_type = 'qwen3_vl_moe'
    model_cls = 'transformers:Qwen3VLMoeForConditionalGeneration'
    architectures = ['Qwen3VLMoeForConditionalGeneration']
    is_moe = True
    models = [
        'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'Qwen/Qwen3-VL-30B-A3B-Thinking',
        'Qwen/Qwen3-VL-30B-A3B-Instruct-FP8',
        'Qwen/Qwen3-VL-30B-A3B-Thinking-FP8',
        'Qwen/Qwen3-VL-235B-A22B-Instruct',
        'Qwen/Qwen3-VL-235B-A22B-Thinking',
        'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8',
        'Qwen/Qwen3-VL-235B-A22B-Thinking-FP8',
    ]


@register_model
class Qwen3_5Loader(Qwen3VLLoader):

    model_type = 'qwen3_5'
    model_cls = 'transformers:Qwen3_5ForConditionalGeneration'
    architectures = ['Qwen3_5ForConditionalGeneration']
    template = 'qwen3_5'
    requires = ['transformers>=5.0.0.dev', 'qwen_vl_utils>=0.0.14', 'decord']
    models = [
        'Qwen/Qwen3.5-0.8B',
        'Qwen/Qwen3.5-2B',
        'Qwen/Qwen3.5-4B',
        'Qwen/Qwen3.5-9B',
        'Qwen/Qwen3.5-27B',
        'Qwen/Qwen3.5-27B-FP8',
        'Qwen/Qwen3.5-0.8B-Base',
        'Qwen/Qwen3.5-2B-Base',
        'Qwen/Qwen3.5-4B-Base',
        'Qwen/Qwen3.5-9B-Base',
        'Qwen/Qwen3.6-27B',
        'Qwen/Qwen3.6-27B-FP8',
    ]

    @property
    def model_arch(self) -> ModelArch:
        # PORT-9: qwen3_5 uses the qwen2_vl partition, not qwen3_vl, even though the loader inherits
        # from Qwen3VLLoader. This is also what makes the keep-alive correct for free: the base
        # `process_model` hooks exactly `model_arch.aligner`, which here is the single merger with no
        # deepstack -- so qwen3_5 skips deepstack zeroing without any process_model override (the old
        # PORT-10 "reach past super to skip it" hack is gone).
        return Qwen2VLLoader.model_arch.fget(self)

    def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs) -> PreTrainedModel:
        from swift.model.models.qwen import _patch_qwen3_5_linear_attention_sequence_parallel
        _patch_qwen3_5_linear_attention_sequence_parallel()
        return super().build_model(model_dir, config, processor, **kwargs)


@register_model
class OvisOcr2Loader(Qwen3_5Loader):
    """OvisOCR2: Qwen3.5 loading, an OCR-specific chat template."""

    model_type = 'ovis_ocr2'
    template = 'ovis_ocr2'
    architectures = []
    tags = ['vision']
    models = ['ATH-MaaS/OvisOCR2']


@register_model
class Qwen3_5MoeLoader(Qwen3_5Loader):

    model_type = 'qwen3_5_moe'
    model_cls = 'transformers:Qwen3_5MoeForConditionalGeneration'
    architectures = ['Qwen3_5MoeForConditionalGeneration']
    requires = ['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14', 'decord']
    is_moe = True
    models = [
        'Qwen/Qwen3.5-35B-A3B-Base',
        'Qwen/Qwen3.5-35B-A3B',
        'Qwen/Qwen3.5-122B-A10B',
        'Qwen/Qwen3.5-397B-A17B',
        'Qwen/Qwen3.5-35B-A3B-FP8',
        'Qwen/Qwen3.5-122B-A10B-FP8',
        'Qwen/Qwen3.5-397B-A17B-FP8',
        'Qwen/Qwen3.6-35B-A3B',
        'Qwen/Qwen3.6-35B-A3B-FP8',
    ]


@register_model
class Qwen3VLEmbLoader(Qwen3VLLoader):
    """Embedding variant: the entire legacy subclass was three env defaults."""

    model_type = 'qwen3_vl_emb'
    template = 'qwen3_vl_emb'
    # Unlike the template variants above, embedding is a different *task*: falling through to
    # `qwen3_vl` (generation) would be wrong, so we keep the architecture declared. Reverse-lookup
    # then returns qwen3_vl / qwen3_vl_emb / qwen3_vl_reranker together and the caller must pick.
    architectures = ['Qwen3VLForConditionalGeneration']
    mcore_model_type = 'qwen3_vl'
    env_defaults = {'IMAGE_MAX_TOKEN_NUM': '1800', 'FPS': '1', 'FPS_MAX_FRAMES': '64'}
    models = ['Qwen/Qwen3-VL-Embedding-2B', 'Qwen/Qwen3-VL-Embedding-8B']


@register_model
class Qwen3VLRerankerLoader(Qwen3VLLoader):

    model_type = 'qwen3_vl_reranker'
    template = 'qwen3_vl_reranker'
    architectures = ['Qwen3VLForConditionalGeneration']  # see Qwen3VLEmbLoader: distinct task, keep it
    mcore_model_type = 'qwen3_vl'
    env_defaults = {'IMAGE_MAX_TOKEN_NUM': '1280', 'FPS': '1', 'FPS_MAX_FRAMES': '64'}
    models = ['Qwen/Qwen3-VL-Reranker-2B', 'Qwen/Qwen3-VL-Reranker-8B']

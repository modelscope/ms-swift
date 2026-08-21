# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the Moonshot Kimi multimodal families (from ``swift/model/models/moonshot.py``).

Only ``kimi_k3`` is migrated: remote-code ``KimiK3ForConditionalGeneration`` (no in-tree class on
transformers 5.5) loaded via ``AutoModelForImageTextToText`` + the ``trust_remote_code`` flag. Its sole
legacy seam is silencing a chatty remote-code tokenizer logger, which fits ``build_processor``.

Not migrated here (see MODEL_MIGRATION.md):
  * ``kimi_vl`` -- pinned ``transformers<4.49`` (dead on dev's 5.5); its loader also deletes
    ``_supports_sdpa`` off the dynamic class and applies ``patch_get_input_embeddings``, the latter
    obsolete per PATCH_INVENTORY.
  * ``kimi_k25`` -- pinned ``transformers>=4.57.1,<5.0.0``; the ``<5.0.0`` ceiling excludes dev's 5.5,
    so the family is version-dead even though it needs no loader logic at all.
"""
from __future__ import annotations

from .base import ModelArch, ModelLoader, register_model


@register_model
class KimiK3Loader(ModelLoader):

    model_type = 'kimi_k3'
    model_cls = 'transformers:AutoModelForImageTextToText'
    trust_remote_code = True
    architectures = ['KimiK3ForConditionalGeneration']
    template = 'kimi_k3'
    requires = ['transformers>=5', 'tiktoken']
    tags = ['vision']
    is_multimodal = True
    models = [('moonshotai/Kimi-K3', 'moonshotai/Kimi-K3')]

    @property
    def model_arch(self) -> ModelArch:
        # legacy `ModelArch.kimi_k25`, shared with the (version-dead) K2.5 family.
        return ModelArch(
            language_model='language_model',
            aligner='mm_projector',
            vision_tower='vision_tower',
        )

    def build_processor(self, model_dir, config, **kwargs):
        processor = super().build_processor(model_dir, config, **kwargs)
        # The remote-code tokenizer (tokenization_kimi.py) warns on every
        # `encode(..., add_special_tokens=False)` call, which spams streaming inference; silence it.
        import logging
        tokenizer = getattr(processor, 'tokenizer', processor)
        logging.getLogger(type(tokenizer).__module__).setLevel(logging.ERROR)
        return processor

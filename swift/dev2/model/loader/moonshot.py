# Copyright (c) ModelScope Contributors. All rights reserved.
# Only kimi_k3 migrated; kimi_vl / kimi_k25 are version-dead on transformers 5.5 (MODEL_MIGRATION.md).
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

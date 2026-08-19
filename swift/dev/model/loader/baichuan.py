# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loader for Baichuan-M1.

Ported from ``swift/model/models/baichuan.py``. Only ``baichuan_m1`` (2025) is migrated.

Dropped (see MODEL_MIGRATION.md):
  * ``baichuan`` / ``baichuan2`` -- 2023 checkpoints pinned to ``transformers<4.34``; ``baichuan2``
    further depends on ``patch_baichuan2_lm_head_forward``, which PATCH_INVENTORY.md marks 不迁移.
"""
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class BaichuanM1Loader(ModelLoader):

    model_type = 'baichuan_m1'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['BaichuanM1ForCausalLM']
    template = 'baichuan_m1'
    requires = ['transformers>=4.48']
    models = [('baichuan-inc/Baichuan-M1-14B-Instruct', 'baichuan-inc/Baichuan-M1-14B-Instruct')]

    def build_model(self, model_dir, config, processor, **kwargs):
        # The remote RotaryEmbedding computes in q's dtype; cast q to k's dtype first to avoid a
        # dtype mismatch. Patch the dynamic-module class before the weights are instantiated.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        rotary_embedding = get_class_from_dynamic_module('modeling_baichuan.RotaryEmbedding', model_dir)
        _old_forward = rotary_embedding.forward

        def _new_forward(self, q, k, seqlen_offset=None, cu_seqlens=None, max_seqlen=None):
            q = q.to(k.dtype)
            return _old_forward(self, q, k, seqlen_offset, cu_seqlens, max_seqlen)

        rotary_embedding.forward = _new_forward
        return super().build_model(model_dir, config, processor, **kwargs)

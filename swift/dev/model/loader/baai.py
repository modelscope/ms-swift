# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the BAAI families.

Ported from ``swift/model/models/baai.py``. Only ``bge_reranker`` is migrated, with a pinned
``task_type='reranker'`` (a bge reranker is only ever a reranker; the user need not pass it, and a
user ``--task_type`` still overrides). Dropped (see MODEL_MIGRATION.md): ``emu3_gen`` / ``emu3_chat``
git-clone the upstream Emu3 repo and download a separate VisionTokenizer to assemble the processor --
2024 constructions too complex to carry forward.
"""
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class BgeRerankerLoader(ModelLoader):

    model_type = 'bge_reranker'
    model_cls = 'transformers:AutoModelForSequenceClassification'
    architectures = ['XLMRobertaForSequenceClassification']
    template = 'bge_reranker'
    task_type = 'reranker'
    models = [
        'BAAI/bge-reranker-base',
        'BAAI/bge-reranker-v2-m3',
        'BAAI/bge-reranker-large',
    ]

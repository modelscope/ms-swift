# Copyright (c) ModelScope Contributors. All rights reserved.
# Only bge_reranker migrated; emu3_gen/chat dropped (2024 git-clone constructions, MODEL_MIGRATION.md).
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

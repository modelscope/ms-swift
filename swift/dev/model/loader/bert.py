# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the BERT families.

Ported from ``swift/model/models/bert.py``. These are encoder backbones: the training task
(seq_cls with ``num_labels`` / embedding) is the user's choice via ``--task_type``, not a fixed
property of the checkpoint -- swift has no ``mlm`` task, so a ``*ForMaskedLM`` checkpoint is just a
seq_cls/embedding backbone. The gte checkpoints, which are only ever embedding / reranker, pin that
task. The legacy embedding hook (CLS-pool + L2 normalize) is NOT reproduced here: dev does
pooling/normalization downstream in the loss/processor layer (see PATCH_INVENTORY.md
``patch_output_normalizer``).
"""
from __future__ import annotations

from .base import ModelLoader, register_model


@register_model
class ModernBertLoader(ModelLoader):
    """A ModernBERT encoder backbone; trained as seq_cls/embedding per the user's ``--task_type``."""

    model_type = 'modern_bert'
    model_cls = 'transformers:AutoModel'
    architectures = ['ModernBertForMaskedLM']
    template = 'dummy'
    requires = ['transformers>=4.48']
    tags = ['bert']
    models = [
        ('answerdotai/ModernBERT-base', 'answerdotai/ModernBERT-base'),
        ('answerdotai/ModernBERT-large', 'answerdotai/ModernBERT-large'),
    ]

    def process_config(self, config):
        # ModernBert's compiled path breaks under some train/quantize setups; legacy disables it.
        config.reference_compile = False
        return config


@register_model
class GteModernBertLoader(ModelLoader):

    model_type = 'modern_bert_gte'
    model_cls = 'transformers:AutoModel'
    architectures = ['ModernBertModel']
    template = 'dummy'
    task_type = 'embedding'
    requires = ['transformers>=4.48']
    tags = ['bert', 'embedding']
    models = [('iic/gte-modernbert-base', 'Alibaba-NLP/gte-modernbert-base')]


@register_model
class GteModernBertRerankerLoader(ModelLoader):

    model_type = 'modern_bert_gte_reranker'
    model_cls = 'transformers:AutoModelForSequenceClassification'
    architectures = ['ModernBertForSequenceClassification']
    template = 'bert'
    task_type = 'reranker'
    requires = ['transformers>=4.48']
    tags = ['bert', 'reranker']
    models = [('iic/gte-reranker-modernbert-base', 'Alibaba-NLP/gte-reranker-modernbert-base')]


@register_model
class BertLoader(ModelLoader):
    """StructBERT backbone; trained as seq_cls/embedding per the user's ``--task_type``."""

    model_type = 'bert'
    model_cls = 'transformers:AutoModel'
    template = 'dummy'
    tags = ['bert']
    models = ['iic/nlp_structbert_backbone_base_std']

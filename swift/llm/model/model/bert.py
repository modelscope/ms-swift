# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType

import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from swift.utils import get_logger
from ..constant import BertModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local, register_model

logger = get_logger()


def get_model_tokenizer_modern_bert(model_dir, *args, **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.reference_compile = False
    kwargs['model_config'] = model_config
    return get_model_tokenizer_from_local(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        BertModelType.modern_bert, [
            ModelGroup([
                Model('answerdotai/ModernBERT-base', 'answerdotai/ModernBERT-base'),
                Model('answerdotai/ModernBERT-large', 'answerdotai/ModernBERT-large'),
            ])
        ],
        None,
        get_model_tokenizer_modern_bert,
        requires=['transformers>=4.48'],
        tags=['bert']))


def get_model_tokenizer_gte_bert(*args, **kwargs):
    kwargs['automodel_class'] = AutoModel
    model, tokenizer = get_model_tokenizer_from_local(*args, **kwargs)
    if model is not None:

        def _normalizer_hook(module, input, output):
            output.last_hidden_state = F.normalize(output.last_hidden_state[:, 0], p=2, dim=1)
            return output

        model.register_forward_hook(_normalizer_hook)
    return model, tokenizer


register_model(
    ModelMeta(
        BertModelType.modern_bert_gte,
        [ModelGroup([
            Model('iic/gte-modernbert-base', 'Alibaba-NLP/gte-modernbert-base'),
        ])],
        None,
        get_model_tokenizer_gte_bert,
        requires=['transformers>=4.48'],
        tags=['bert', 'embedding']))

register_model(
    ModelMeta(
        BertModelType.bert, [ModelGroup([
            Model('iic/nlp_structbert_backbone_base_std'),
        ])],
        None,
        get_model_tokenizer_from_local,
        tags=['bert']))

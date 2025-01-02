# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import AutoConfig

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

register_model(
    ModelMeta(
        BertModelType.bert, [ModelGroup([
            Model('iic/nlp_structbert_backbone_base_std'),
        ])],
        None,
        get_model_tokenizer_from_local,
        tags=['bert']))

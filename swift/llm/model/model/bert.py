# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import AutoConfig

from swift.utils import get_logger
from ..constant import LLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local, register_model

logger = get_logger()


def get_model_tokenizer_modern_bert(model_dir, *args, **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.reference_compile = False
    kwargs['model_config'] = model_config
    return get_model_tokenizer_from_local(model_dir, *args, **kwargs)


register_model(
    ModelMeta(LLMModelType.modern_bert, [
        ModelGroup([
            Model('answerdotai/ModernBERT-base', 'answerdotai/ModernBERT-base'),
            Model('answerdotai/ModernBERT-large', 'answerdotai/ModernBERT-large'),
        ])
    ], None, get_model_tokenizer_from_local), )

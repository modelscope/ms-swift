# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType

import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import BertModelType, RerankerModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()


class ModernBertLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        config.reference_compile = False
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        BertModelType.modern_bert, [
            ModelGroup([
                Model('answerdotai/ModernBERT-base', 'answerdotai/ModernBERT-base'),
                Model('answerdotai/ModernBERT-large', 'answerdotai/ModernBERT-large'),
            ])
        ],
        ModernBertLoader,
        requires=['transformers>=4.48'],
        tags=['bert']))


class GTEBertLoader(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        self.automodel_class = self.automodel_class or AutoModel
        model = super().get_model(model_dir, config, model_kwargs)

        def _normalizer_hook(module, input, output):
            output.last_hidden_state = F.normalize(output.last_hidden_state[:, 0], p=2, dim=1)
            return output

        model.register_forward_hook(_normalizer_hook)
        return model


register_model(
    ModelMeta(
        BertModelType.modern_bert_gte,
        [ModelGroup([
            Model('iic/gte-modernbert-base', 'Alibaba-NLP/gte-modernbert-base'),
        ])],
        GTEBertLoader,
        requires=['transformers>=4.48'],
        tags=['bert', 'embedding']))


class GTEBertReranker(ModelLoader):

    def get_model(self, model_dir: str, config, model_kwargs) -> PreTrainedModel:
        self.automodel_class = self.automodel_class or AutoModelForSequenceClassification
        return super().get_model(model_dir, config, model_kwargs)


register_model(
    ModelMeta(
        RerankerModelType.modern_bert_gte_reranker,
        [ModelGroup([
            Model('iic/gte-reranker-modernbert-base', 'Alibaba-NLP/gte-reranker-modernbert-base'),
        ])],
        GTEBertReranker,
        template=TemplateType.bert,
        requires=['transformers>=4.48'],
        tags=['bert', 'reranker']))

register_model(
    ModelMeta(BertModelType.bert, [ModelGroup([
        Model('iic/nlp_structbert_backbone_base_std'),
    ])], tags=['bert']))

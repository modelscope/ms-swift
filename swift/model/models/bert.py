# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel

from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import BertModelType, LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()


class ModernBertLoader(ModelLoader):

    def get_model(self, model_dir: str, config, *args, **kwargs) -> PreTrainedModel:
        config.reference_compile = False
        return super().get_model(model_dir, config, *args, **kwargs)


register_model(
    ModelMeta(
        BertModelType.modern_bert, [
            ModelGroup([
                Model('answerdotai/ModernBERT-base', 'answerdotai/ModernBERT-base'),
                Model('answerdotai/ModernBERT-large', 'answerdotai/ModernBERT-large'),
            ])
        ],
        ModernBertLoader,
        template=TemplateType.dummy,
        requires=['transformers>=4.48'],
        architectures=['ModernBertForMaskedLM'],
        tags=['bert']))


class GTEBertLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        self.auto_model_cls = self.auto_model_cls or AutoModel
        model = super().get_model(model_dir, *args, **kwargs)

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
        template=TemplateType.dummy,
        requires=['transformers>=4.48'],
        architectures=['ModernBertModel'],
        tags=['bert', 'embedding']))


class GTEBertReranker(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        self.auto_model_cls = self.auto_model_cls or AutoModelForSequenceClassification
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.modern_bert_gte_reranker,
        [ModelGroup([
            Model('iic/gte-reranker-modernbert-base', 'Alibaba-NLP/gte-reranker-modernbert-base'),
        ])],
        GTEBertReranker,
        template=TemplateType.bert,
        requires=['transformers>=4.48'],
        architectures=['ModernBertForSequenceClassification'],
        task_type='reranker',
        tags=['bert', 'reranker']))

register_model(
    ModelMeta(
        BertModelType.bert, [ModelGroup([
            Model('iic/nlp_structbert_backbone_base_std'),
        ])],
        template=TemplateType.dummy,
        tags=['bert']))

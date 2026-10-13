# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import AutoModel, PreTrainedModel
from types import MethodType

from swift.template import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


def patch_jina_reranker_v3_forward(model: PreTrainedModel) -> PreTrainedModel:
    if hasattr(model, '_swift_forward_origin'):
        return model
    model._swift_forward_origin = model.forward

    def forward(self, *args, **kwargs):
        kwargs.pop('labels', None)
        output = self._swift_forward_origin(*args, **kwargs)
        logits = output.scores
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        output.logits = logits
        return output

    model.forward = MethodType(forward, model)
    return model


class JinaRerankerV3Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        self.auto_model_cls = self.auto_model_cls or AutoModel
        # Jina ships its own ranking projector in remote code. Avoid the generic
        # reranker patch, which would replace that head with a new classifier.
        task_type = self.model_info.task_type
        self.model_info.task_type = 'causal_lm'
        try:
            model = super().get_model(model_dir, *args, **kwargs)
        finally:
            self.model_info.task_type = task_type
        return patch_jina_reranker_v3_forward(model)


register_model(
    ModelMeta(
        LLMModelType.jina_reranker_v3,
        [
            ModelGroup([
                Model(None, 'jinaai/jina-reranker-v3'),
                Model(None, 'jinaai/jina-reranker-v3.5'),
            ]),
        ],
        JinaRerankerV3Loader,
        template=TemplateType.jina_reranker_v3,
        model_arch=ModelArch.llama,
        architectures=['JinaForRanking'],
        task_type='reranker',
        requires=['transformers>=4.55'],
        tags=['reranker'],
    ))

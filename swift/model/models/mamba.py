# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import PreTrainedModel

from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()


class MambaLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        logger.info(
            '[IMPORTANT] Remember installing causal-conv1d>=1.2.0 and mamba-ssm, or you training and inference will'
            'be really slow!')
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.mamba,
        [
            ModelGroup([
                Model('AI-ModelScope/mamba-130m-hf', 'state-spaces/mamba-130m-hf'),
                Model('AI-ModelScope/mamba-370m-hf', 'state-spaces/mamba-370m-hf'),
                Model('AI-ModelScope/mamba-390m-hf', 'state-spaces/mamba-390m-hf'),
                Model('AI-ModelScope/mamba-790m-hf', 'state-spaces/mamba-790m-hf'),
                Model('AI-ModelScope/mamba-1.4b-hf', 'state-spaces/mamba-1.4b-hf'),
                Model('AI-ModelScope/mamba-2.8b-hf', 'state-spaces/mamba-2.8b-hf'),
            ])
        ],
        MambaLoader,
        template=TemplateType.default,
        architectures=['MambaForCausalLM'],
        model_arch=None,
        requires=['transformers>=4.39.0'],
    ))

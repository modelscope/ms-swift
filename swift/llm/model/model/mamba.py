# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local, register_model
from ..utils import ModelInfo

logger = get_logger()


def get_model_tokenizer_mamba(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    logger.info('[IMPORTANT] Remember installing causal-conv1d>=1.2.0 and mamba-ssm, or you training and inference will'
                'be really slow!')
    return get_model_tokenizer_from_local(model_dir, model_info, model_kwargs, load_model, **kwargs)


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
        TemplateType.default,
        get_model_tokenizer_mamba,
        architectures=['MambaForCausalLM'],
        model_arch=None,
        requires=['transformers>=4.39.0'],
    ))

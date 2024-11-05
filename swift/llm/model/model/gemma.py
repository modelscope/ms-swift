# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers import PretrainedConfig

from swift.llm import TemplateType
from ..constant import LLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local, register_model


def get_model_tokenizer_paligemma_vision(model_dir: str,
                                         model_config: PretrainedConfig,
                                         model_kwargs: Dict[str, Any],
                                         load_model: bool = True,
                                         **kwargs):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_from_local(
        model_dir, model_config, model_kwargs, load_model, automodel_class=PaliGemmaForConditionalGeneration, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.mamba,
        [
            # llama2
            ModelGroup([
                Model('AI-ModelScope/paligemma-3b-pt-224', 'google/paligemma-3b-pt-224'),
                Model('AI-ModelScope/paligemma-3b-pt-448', 'google/paligemma-3b-pt-448'),
                Model('AI-ModelScope/paligemma-3b-pt-896', 'google/paligemma-3b-pt-896'),
                Model('AI-ModelScope/paligemma-3b-mix-448', 'google/paligemma-3b-mix-448'),
            ],
                       requires=['transformers>=4.41'],
                       tags=['multi-modal', 'vision'],
                       ignore_file_pattern=[r'.+\.bin$']),
        ],
        TemplateType.paligemma,
        get_model_tokenizer_paligemma_vision,
        architectures=['MambaForCausalLM'],
        support_flash_attn=True,
        support_vllm=True,
    ))

# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model

logger = get_logger()

register_model(
    ModelMeta(
        LLMModelType.openbuddy_llama,
        [
            ModelGroup([
                Model('OpenBuddy/openbuddy-llama-65b-v8-bf16', 'OpenBuddy/openbuddy-llama-65b-v8-bf16'),
            ]),
            ModelGroup([
                Model('OpenBuddy/openbuddy-llama2-13b-v8.1-fp16', 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16'),
                Model('OpenBuddy/openbuddy-llama2-70b-v10.1-bf16', 'OpenBuddy/openbuddy-llama2-70b-v10.1-bf16'),
            ]),
            ModelGroup([
                Model('OpenBuddy/openbuddy-deepseek-67b-v15.2', 'OpenBuddy/openbuddy-deepseek-67b-v15.2'),
            ]),
        ],
        TemplateType.openbuddy,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_llama3,
        [
            ModelGroup([
                Model('OpenBuddy/openbuddy-llama3-8b-v21.1-8k', 'OpenBuddy/openbuddy-llama3-8b-v21.1-8k'),
                Model('OpenBuddy/openbuddy-llama3-70b-v21.1-8k', 'OpenBuddy/openbuddy-llama3-70b-v21.1-8k'),
                Model('OpenBuddy/openbuddy-yi1.5-34b-v21.3-32k', 'OpenBuddy/openbuddy-yi1.5-34b-v21.3-32k'),
            ]),
            ModelGroup([
                Model('OpenBuddy/openbuddy-llama3.1-8b-v22.1-131k', 'OpenBuddy/openbuddy-llama3.1-8b-v22.1-131k'),
                Model('OpenBuddy/openbuddy-nemotron-70b-v23.2-131k', 'OpenBuddy/openbuddy-nemotron-70b-v23.2-131k'),
            ],
                       requires=['transformers>=4.43']),
            ModelGroup(
                [Model('OpenBuddy/openbuddy-llama3.3-70b-v24.3-131k', 'OpenBuddy/openbuddy-llama3.3-70b-v24.3-131k')],
                requires=['transformers>=4.45'])
        ],
        TemplateType.openbuddy2,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_mistral,
        [
            ModelGroup([
                Model('OpenBuddy/openbuddy-mistral-7b-v17.1-32k', 'OpenBuddy/openbuddy-mistral-7b-v17.1-32k'),
            ]),
            ModelGroup([
                Model('OpenBuddy/openbuddy-zephyr-7b-v14.1', 'OpenBuddy/openbuddy-zephyr-7b-v14.1'),
            ]),
        ],
        TemplateType.openbuddy,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        requires=['transformers>=4.34'],
        architectures=['MistralForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_mixtral,
        [
            ModelGroup([
                Model('OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k', 'OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k'),
            ], ),
        ],
        TemplateType.openbuddy,
        get_model_tokenizer_with_flash_attn,
        architectures=['MixtralForCausalLM'],
        requires=['transformers>=4.36'],
    ))

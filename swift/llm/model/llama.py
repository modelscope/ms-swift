from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoTokenizer

from swift.llm import TemplateType
from .constant import LLMModelType, MLLMModelType
from .register import (Model, ModelGroup, TemplateGroup, get_model_tokenizer_multimodal,
                       get_model_tokenizer_with_flash_attn, register_model)


def get_model_tokenizer_llama2(model_dir: str,
                               torch_dtype: torch.dtype,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    return get_model_tokenizer_with_flash_attn(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


register_model(
    LLMModelType.llama2,
    'LlamaForCausalLM',
    [
        # llama2
        ModelGroup([
            # base
            Model('modelscope/Llama-2-7b-ms', 'meta-llama/Llama-2-7b-hf'),
            Model('modelscope/Llama-2-13b-ms', 'meta-llama/Llama-2-13b-hf'),
            Model('modelscope/Llama-2-70b-ms', 'meta-llama/Llama-2-70b-hf'),
            # chat
            Model('modelscope/Llama-2-7b-chat-ms', 'meta-llama/Llama-2-7b-chat-hf'),
            Model('modelscope/Llama-2-13b-chat-ms', 'meta-llama/Llama-2-13b-chat-hf'),
            Model('modelscope/Llama-2-70b-chat-ms', 'meta-llama/Llama-2-70b-chat-hf'),
        ]),
        # chinese-llama2
        ModelGroup([
            # base
            Model('AI-ModelScope/chinese-llama-2-1.3b', 'hfl/chinese-llama-2-1.3b'),
            Model('AI-ModelScope/chinese-llama-2-7b', 'hfl/chinese-llama-2-7b'),
            Model('AI-ModelScope/chinese-llama-2-7b-16k', 'hfl/chinese-llama-2-7b-16k'),
            Model('AI-ModelScope/chinese-llama-2-7b-64k', 'hfl/chinese-llama-2-7b-64k'),
            Model('AI-ModelScope/chinese-llama-2-13b', 'hfl/chinese-llama-2-13b'),
            Model('AI-ModelScope/chinese-llama-2-13b-16k', 'hfl/chinese-llama-2-13b-16k'),
            # chat
            Model('AI-ModelScope/chinese-alpaca-2-1.3b', 'hfl/chinese-alpaca-2-1.3b'),
            Model('AI-ModelScope/chinese-alpaca-2-7b', 'hfl/chinese-alpaca-2-7b'),
            Model('AI-ModelScope/chinese-alpaca-2-7b-16k', 'hfl/chinese-alpaca-2-7b-16k'),
            Model('AI-ModelScope/chinese-alpaca-2-7b-64k', 'hfl/chinese-alpaca-2-7b-64k'),
            Model('AI-ModelScope/chinese-alpaca-2-13b', 'hfl/chinese-alpaca-2-13b'),
            Model('AI-ModelScope/chinese-alpaca-2-13b-16k', 'hfl/chinese-alpaca-2-13b-16k'),
        ]),
        # base quant
        ModelGroup([
            Model('AI-ModelScope/Llama-2-7b-AQLM-2Bit-1x16-hf', 'ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf'),
        ])
    ],
    TemplateGroup(TemplateType.llama),
    get_model_tokenizer_llama2,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
)

register_model(
    LLMModelType.llama3,
    'LlamaForCausalLM',
    [
        # llama3
        ModelGroup([
            # base
            Model('LLM-Research/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B'),
            Model('LLM-Research/Meta-Llama-3-70B', 'meta-llama/Meta-Llama-3-70B'),
            # chat
            Model('LLM-Research/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'),
            Model('LLM-Research/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-70B-Instruct'),
        ]),
        # llama3-quant
        ModelGroup([
            Model('swift/Meta-Llama-3-8B-Instruct-GPTQ-Int4', 'study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int4'),
            Model('swift/Meta-Llama-3-8B-Instruct-GPTQ-Int8', 'study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int8'),
            Model('swift/Meta-Llama-3-8B-Instruct-AWQ', 'study-hjt/Meta-Llama-3-8B-Instruct-AWQ'),
            Model('swift/Meta-Llama-3-70B-Instruct-GPTQ-Int4', 'study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int4'),
            Model('swift/Meta-Llama-3-70B-Instruct-GPTQ-Int8', 'study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8'),
            Model('swift/Meta-Llama-3-70B-Instruct-AWQ', 'study-hjt/Meta-Llama-3-70B-Instruct-AWQ'),
        ]),
        # chinese-llama3
        ModelGroup([
            Model('ChineseAlpacaGroup/llama-3-chinese-8b', 'hfl/llama-3-chinese-8b'),
            Model('ChineseAlpacaGroup/llama-3-chinese-8b-instruct', 'hfl/llama-3-chinese-8b-instruct'),
        ]),
    ],
    TemplateGroup(TemplateType.llama3),
    get_model_tokenizer_with_flash_attn,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
)

register_model(
    LLMModelType.llama3_1,
    'LlamaForCausalLM',
    [
        # llama3.1
        ModelGroup([
            # base
            Model('LLM-Research/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B'),
            Model('LLM-Research/Meta-Llama-3.1-70B', 'meta-llama/Meta-Llama-3.1-70B'),
            Model('LLM-Research/Meta-Llama-3.1-405B', 'meta-llama/Meta-Llama-3.1-405B'),
            # chat
            Model('LLM-Research/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
            Model('LLM-Research/Meta-Llama-3.1-70B-Instruct', 'meta-llama/Meta-Llama-3.1-70B-Instruct'),
            Model('LLM-Research/Meta-Llama-3.1-405B-Instruct', 'meta-llama/Meta-Llama-3.1-405B-Instruct'),
            # fp8
            Model('LLM-Research/Meta-Llama-3.1-70B-Instruct-FP8', 'meta-llama/Meta-Llama-3.1-70B-Instruct-FP8'),
            Model('LLM-Research/Meta-Llama-3.1-405B-Instruct-FP8', 'meta-llama/Meta-Llama-3.1-405B-Instruct-FP8'),
        ]),
        # llama3.1-quant
        ModelGroup([
            # bnb-nf4
            Model('LLM-Research/Meta-Llama-3.1-8B-Instruct-BNB-NF4',
                  'hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4'),
            Model('LLM-Research/Meta-Llama-3.1-70B-Instruct-bnb-4bit', 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit'),
            Model('LLM-Research/Meta-Llama-3.1-405B-Instruct-BNB-NF4',
                  'hugging-quants/Meta-Llama-3.1-405B-Instruct-BNB-NF4'),
            # gptq-int4
            Model('LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4',
                  'hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4'),
            Model('LLM-Research/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4',
                  'hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4'),
            Model('LLM-Research/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4',
                  'hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4'),
            # awq-int4
            Model('LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
                  'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4'),
            Model('LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
                  'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4'),
            Model('LLM-Research/Meta-Llama-3.1-405B-Instruct-AWQ-INT4',
                  'hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4'),
        ]),
    ],
    TemplateGroup(TemplateType.llama3),
    get_model_tokenizer_with_flash_attn,
    requires=['transformers>=4.43'],
    ignore_file_pattern=[r'.+\.pth$'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
)

register_model(
    LLMModelType.llama3_2,
    'LlamaForCausalLM',
    [
        ModelGroup([
            Model('LLM-Research/Llama-3.2-1B', 'meta-llama/Llama-3.2-1B'),
            Model('LLM-Research/Llama-3.2-3B', 'meta-llama/Llama-3.2-3B'),
            Model('LLM-Research/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct'),
            Model('LLM-Research/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct'),
        ])
    ],
    TemplateGroup(TemplateType.llama3_2),
    get_model_tokenizer_with_flash_attn,
    requires=['transformers>=4.45'],
    ignore_file_pattern=[r'.+\.pth$'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
)

register_model(
    LLMModelType.longwriter_llama3_1,
    'LlamaForCausalLM',
    [ModelGroup([
        Model('ZhipuAI/LongWriter-llama3.1-8b', 'THUDM/LongWriter-llama3.1-8b'),
    ])],
    TemplateGroup(TemplateType.longwriter_llama3),
    get_model_tokenizer_with_flash_attn,
    requires=['transformers>=4.43'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
)


def get_model_tokenizer_yi(model_dir, *args, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    return get_model_tokenizer_with_flash_attn(model_dir, *args, tokenizer=tokenizer, **kwargs)


register_model(
    LLMModelType.yi,
    'LlamaForCausalLM',
    # yi
    [
        ModelGroup([
            Model('01ai/Yi-6B', '01-ai/Yi-6B'),
            Model('01ai/Yi-6B-200K', '01-ai/Yi-6B-200K'),
            Model('01ai/Yi-6B-Chat', '01-ai/Yi-6B-Chat'),
            Model('01ai/Yi-6B-Chat-4bits', '01-ai/Yi-6B-Chat-4bits'),
            Model('01ai/Yi-6B-Chat-8bits', '01-ai/Yi-6B-Chat-8bits'),
            Model('01ai/Yi-9B', '01-ai/Yi-9B'),
            Model('01ai/Yi-9B-200K', '01-ai/Yi-9B-200K'),
            Model('01ai/Yi-34B', '01-ai/Yi-34B'),
            Model('01ai/Yi-34B-200K', '01-ai/Yi-34B-200K'),
            Model('01ai/Yi-34B-Chat', '01-ai/Yi-34B-Chat'),
            Model('01ai/Yi-34B-Chat-4bits', '01-ai/Yi-34B-Chat-4bits'),
            Model('01ai/Yi-34B-Chat-8bits', '01-ai/Yi-34B-Chat-8bits'),
        ]),
        # yi1.5
        ModelGroup([
            Model('01ai/Yi-1.5-6B', '01-ai/Yi-1.5-6B'),
            Model('01ai/Yi-1.5-6B-Chat', '01-ai/Yi-1.5-6B-Chat'),
            Model('01ai/Yi-1.5-9B', '01-ai/Yi-1.5-9B'),
            Model('01ai/Yi-1.5-9B-Chat', '01-ai/Yi-1.5-9B-Chat'),
            Model('01ai/Yi-1.5-9B-Chat-16K', '01-ai/Yi-1.5-9B-Chat-16K'),
            Model('01ai/Yi-1.5-34B', '01-ai/Yi-1.5-34B'),
            Model('01ai/Yi-1.5-34B-Chat', '01-ai/Yi-1.5-34B-Chat'),
            Model('01ai/Yi-1.5-34B-Chat-16K', '01-ai/Yi-1.5-34B-Chat-16K'),
        ]),
        # yi1.5-quant
        ModelGroup([
            Model('AI-ModelScope/Yi-1.5-6B-Chat-GPTQ', 'modelscope/Yi-1.5-6B-Chat-GPTQ'),
            Model('AI-ModelScope/Yi-1.5-6B-Chat-AWQ', 'modelscope/Yi-1.5-6B-Chat-AWQ'),
            Model('AI-ModelScope/Yi-1.5-9B-Chat-GPTQ', 'modelscope/Yi-1.5-9B-Chat-GPTQ'),
            Model('AI-ModelScope/Yi-1.5-9B-Chat-AWQ', 'modelscope/Yi-1.5-9B-Chat-AWQ'),
            Model('AI-ModelScope/Yi-1.5-34B-Chat-GPTQ', 'modelscope/Yi-1.5-34B-Chat-GPTQ'),
            Model('AI-ModelScope/Yi-1.5-34B-Chat-AWQ', 'modelscope/Yi-1.5-34B-Chat-AWQ'),
        ]),
    ],
    TemplateGroup(TemplateType.chatml),
    get_model_tokenizer_yi,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
)

register_model(
    LLMModelType.yi_coder,
    'LlamaForCausalLM',
    [
        ModelGroup([
            Model('01ai/Yi-Coder-1.5B', '01-ai/Yi-Coder-1.5B'),
            Model('01ai/Yi-Coder-9B', '01-ai/Yi-Coder-9B'),
            Model('01ai/Yi-Coder-1.5B-Chat', '01-ai/Yi-Coder-1.5B-Chat'),
            Model('01ai/Yi-Coder-9B-Chat', '01-ai/Yi-Coder-9B-Chat'),
        ],
                   tags=['coding'])
    ],
    TemplateGroup(TemplateType.yi_coder),
    get_model_tokenizer_with_flash_attn,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
)


def get_model_tokenizer_llama3_2_vision(*args, **kwargs):
    from transformers import MllamaForConditionalGeneration
    kwargs['automodel_class'] = MllamaForConditionalGeneration
    return get_model_tokenizer_multimodal(*args, **kwargs)


register_model(
    MLLMModelType.llama3_2_vision,
    'MllamaForConditionalGeneration',
    [
        ModelGroup([
            Model('LLM-Research/Llama-3.2-11B-Vision', 'meta-llama/Llama-3.2-11B-Vision'),
            Model('LLM-Research/Llama-3.2-90B-Vision', 'meta-llama/Llama-3.2-90B-Vision'),
            Model('LLM-Research/Llama-3.2-11B-Vision-Instruct', 'meta-llama/Llama-3.2-11B-Vision-Instruct'),
            Model('LLM-Research/Llama-3.2-90B-Vision-Instruct', 'meta-llama/Llama-3.2-90B-Vision-Instruct'),
        ],
                   tags=['vision'])
    ],
    TemplateGroup(TemplateType.llama3_2_vision, TemplateType.llama3_2_vision_generation),
    get_model_tokenizer_llama3_2_vision,
    requires=['transformers>=4.45'],
    ignore_file_pattern=['*.pth'],
    is_multimodal=True,
    support_flash_attn=True,
    support_vllm=True,
)

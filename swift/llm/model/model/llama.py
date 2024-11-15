# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict

from modelscope import AutoConfig

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, git_clone_github


def get_model_tokenizer_llama(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    kwargs['model_config'] = model_config
    return get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.llama,
        [
            # llama2
            ModelGroup(
                [
                    # base
                    Model('modelscope/Llama-2-7b-ms', 'meta-llama/Llama-2-7b-hf'),
                    Model('modelscope/Llama-2-13b-ms', 'meta-llama/Llama-2-13b-hf'),
                    Model('modelscope/Llama-2-70b-ms', 'meta-llama/Llama-2-70b-hf'),
                    # chat
                    Model('modelscope/Llama-2-7b-chat-ms', 'meta-llama/Llama-2-7b-chat-hf'),
                    Model('modelscope/Llama-2-13b-chat-ms', 'meta-llama/Llama-2-13b-chat-hf'),
                    Model('modelscope/Llama-2-70b-chat-ms', 'meta-llama/Llama-2-70b-chat-hf'),
                ],
                ignore_file_pattern=[r'.+\.bin$']),
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
            ]),
        ],
        TemplateType.llama,
        get_model_tokenizer_llama,
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        LLMModelType.llama3,
        [
            # llama3
            ModelGroup(
                [
                    # base
                    Model('LLM-Research/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B'),
                    Model('LLM-Research/Meta-Llama-3-70B', 'meta-llama/Meta-Llama-3-70B'),
                    # chat
                    Model('LLM-Research/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'),
                    Model('LLM-Research/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-70B-Instruct'),
                ],
                TemplateType.llama3),
            # llama3-quant
            ModelGroup([
                Model('swift/Meta-Llama-3-8B-Instruct-GPTQ-Int4', 'study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int4'),
                Model('swift/Meta-Llama-3-8B-Instruct-GPTQ-Int8', 'study-hjt/Meta-Llama-3-8B-Instruct-GPTQ-Int8'),
                Model('swift/Meta-Llama-3-8B-Instruct-AWQ', 'study-hjt/Meta-Llama-3-8B-Instruct-AWQ'),
                Model('swift/Meta-Llama-3-70B-Instruct-GPTQ-Int4', 'study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int4'),
                Model('swift/Meta-Llama-3-70B-Instruct-GPTQ-Int8', 'study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8'),
                Model('swift/Meta-Llama-3-70B-Instruct-AWQ', 'study-hjt/Meta-Llama-3-70B-Instruct-AWQ'),
            ], TemplateType.llama3),
            # chinese-llama3
            ModelGroup([
                Model('ChineseAlpacaGroup/llama-3-chinese-8b', 'hfl/llama-3-chinese-8b'),
                Model('ChineseAlpacaGroup/llama-3-chinese-8b-instruct', 'hfl/llama-3-chinese-8b-instruct'),
            ], TemplateType.llama3),
            # llama3.1
            ModelGroup(
                [
                    # base
                    Model('LLM-Research/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B'),
                    Model('LLM-Research/Meta-Llama-3.1-70B', 'meta-llama/Meta-Llama-3.1-70B'),
                    Model('LLM-Research/Meta-Llama-3.1-405B', 'meta-llama/Meta-Llama-3.1-405B'),
                    # chat
                    Model('LLM-Research/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
                    Model('LLM-Research/Meta-Llama-3.1-70B-Instruct', 'meta-llama/Meta-Llama-3.1-70B-Instruct'),
                    Model('AI-ModelScope/Llama-3.1-Nemotron-70B-Instruct-HF',
                          'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'),
                    Model('LLM-Research/Meta-Llama-3.1-405B-Instruct', 'meta-llama/Meta-Llama-3.1-405B-Instruct'),
                    # fp8
                    Model('LLM-Research/Meta-Llama-3.1-70B-Instruct-FP8', 'meta-llama/Meta-Llama-3.1-70B-Instruct-FP8'),
                    Model('LLM-Research/Meta-Llama-3.1-405B-Instruct-FP8',
                          'meta-llama/Meta-Llama-3.1-405B-Instruct-FP8'),
                ],
                TemplateType.llama3,
                ignore_file_pattern=[r'.+\.pth$'],
                requires=['transformers>=4.43']),
            # llama3.1-quant
            ModelGroup(
                [
                    # bnb-nf4
                    Model('LLM-Research/Meta-Llama-3.1-8B-Instruct-BNB-NF4',
                          'hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4'),
                    Model('LLM-Research/Meta-Llama-3.1-70B-Instruct-bnb-4bit',
                          'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit'),
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
                ],
                requires=['transformers>=4.43']),
        ],
        TemplateType.llama3,
        get_model_tokenizer_with_flash_attn,
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        LLMModelType.longwriter_llama3,
        [ModelGroup([
            Model('ZhipuAI/LongWriter-llama3.1-8b', 'THUDM/LongWriter-llama3.1-8b'),
        ])],
        TemplateType.longwriter_llama3,
        get_model_tokenizer_with_flash_attn,
        architectures=['LlamaForCausalLM'],
        requires=['transformers>=4.43'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        LLMModelType.llama3_2,
        [
            ModelGroup([
                Model('LLM-Research/Llama-3.2-1B', 'meta-llama/Llama-3.2-1B'),
                Model('LLM-Research/Llama-3.2-3B', 'meta-llama/Llama-3.2-3B'),
                Model('LLM-Research/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct'),
                Model('LLM-Research/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct'),
            ])
        ],
        TemplateType.llama3_2,
        get_model_tokenizer_with_flash_attn,
        architectures=['LlamaForCausalLM'],
        ignore_file_pattern=[r'.+\.pth$'],
        requires=['transformers>=4.45'],
        model_arch=ModelArch.llama,
    ))


def get_model_tokenizer_llama3_2_vision(*args, **kwargs):
    from transformers import MllamaForConditionalGeneration
    kwargs['automodel_class'] = MllamaForConditionalGeneration
    return get_model_tokenizer_multimodal(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.llama3_2_vision,
        [
            ModelGroup([
                Model('LLM-Research/Llama-3.2-11B-Vision', 'meta-llama/Llama-3.2-11B-Vision'),
                Model('LLM-Research/Llama-3.2-90B-Vision', 'meta-llama/Llama-3.2-90B-Vision'),
                Model('LLM-Research/Llama-3.2-11B-Vision-Instruct', 'meta-llama/Llama-3.2-11B-Vision-Instruct'),
                Model('LLM-Research/Llama-3.2-90B-Vision-Instruct', 'meta-llama/Llama-3.2-90B-Vision-Instruct'),
            ],
                       tags=['vision'])
        ],
        TemplateType.llama3_2_vision,
        get_model_tokenizer_llama3_2_vision,
        requires=['transformers>=4.45'],
        architectures=['MllamaForConditionalGeneration'],
        model_arch=ModelArch.llava3_2_vision,
    ))


def get_model_tokenizer_omnli(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/ictnlp/LLaMA-Omni')
    sys.path.append(os.path.join(local_repo_path))
    from omni_speech.model import OmniSpeech2SLlamaForCausalLM, OmniSpeechLlamaForCausalLM
    import whisper
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.speech_encoder = os.path.join(model_dir, 'large-v3.pt')
    if not os.path.exists(model_config.speech_encoder):
        whisper.load_model('large-v3', download_root=model_dir)
    kwargs['automodel_class'] = OmniSpeech2SLlamaForCausalLM
    kwargs['model_config'] = model_config
    for key in ['forward', 'generate']:
        try:
            delattr(OmniSpeech2SLlamaForCausalLM, key)
            delattr(OmniSpeechLlamaForCausalLM, key)
        except AttributeError:
            pass
    # not support device_map='auto'
    device_map = model_kwargs['device_map']
    model_kwargs['device_map'] = None
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model:
        model.to('cuda:0' if device_map == 'auto' else device_map)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.llama3_1_omni,
        [
            ModelGroup(
                [
                    Model('ICTNLP/Llama-3.1-8B-Omni', 'ICTNLP/Llama-3.1-8B-Omni'),
                ],
                tags=['multi-modal', 'audio', 'skip_test'],
                requires=['whisper', 'openai-whisper'],
            )
        ],
        TemplateType.llama3_1_omni,
        get_model_tokenizer_omnli,
        architectures=['OmniSpeech2SLlamaForCausalLM'],
        model_arch=ModelArch.llama3_1_omni,
    ))

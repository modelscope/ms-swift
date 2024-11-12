# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from functools import partial
from types import MethodType
from typing import Any, Dict, Tuple

import torch
from modelscope import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from ..model_arch import ModelArch
from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType, MLLMModelType
from ..patcher import patch_output_clone
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, use_submodel_func
from .qwen import get_model_tokenizer_qwen

logger = get_logger()


def get_model_tokenizer_grok(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             tokenizer=None,
                             automodel_class=AutoModelForCausalLM,
                             **kwargs):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            'AI-ModelScope/grok-1-tokenizer', revision='master', trust_remote_code=True)
    eos_token = kwargs.get('eos_token')
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    model = None
    if load_model:
        model = automodel_class.from_pretrained(model_dir, trust_remote_code=True, **model_kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.grok, [
            ModelGroup([
                Model('colossalai/grok-1-pytorch', 'hpcai-tech/grok-1'),
            ], ),
        ],
        TemplateType.default,
        get_model_tokenizer_grok,
        architectures=['Grok1ModelForCausalLM'],
        model_arch=ModelArch.llama
        # TODO
    ))


def get_model_tokenizer_mplug_owl3(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    model_cls = get_class_from_dynamic_module('modeling_mplugowl3.mPLUGOwl3Model', model_dir)
    model_cls._no_split_modules = ['SiglipEncoderLayer']
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    processor = model.init_processor(tokenizer)
    tokenizer.processor = processor
    if model is not None:
        func_list = ['generate', 'forward']
        use_submodel_func(model, 'language_model', func_list)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.mplug3, [
            ModelGroup([
                Model('iic/mPLUG-Owl3-1B-241014', 'mPLUG/mPLUG-Owl3-1B-241014'),
                Model('iic/mPLUG-Owl3-2B-241014', 'mPLUG/mPLUG-Owl3-2B-241014'),
                Model('iic/mPLUG-Owl3-7B-240728', 'mPLUG/mPLUG-Owl3-7B-240728'),
            ],
                       requires=['transformers>=4.36', 'icecream'],
                       tags=['multi-modal', 'vision', 'video']),
        ],
        TemplateType.mplug_owl3,
        get_model_tokenizer_mplug_owl3,
        architectures=['mPLUGOwl3Model'],
        model_arch=ModelArch.mplug_owl3))


def get_model_tokenizer_polylm(model_dir: str,
                               model_info: ModelInfo,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=True)
    return get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.polylm,
        [
            ModelGroup(
                [
                    # base
                    Model('damo/nlp_polylm_13b_text_generation', 'DAMO-NLP-MT/polylm-13b'),
                ], ),
        ],
        TemplateType.default,
        get_model_tokenizer_polylm,
        architectures=['GPT2LMHeadModel'],
        model_arch=ModelArch.qwen))


def get_skywork_model_tokenizer(model_dir: str,
                                model_info: ModelInfo,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if 'chat' in model_dir:
        tokenizer.add_tokens('[USER]')
        tokenizer.add_tokens('[BOT]')
        tokenizer.add_tokens('[SEP]')
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.skywork,
        [
            ModelGroup([
                Model('skywork/Skywork-13B-chat'),
                Model('skywork/Skywork-13B-base'),
            ]),
        ],
        TemplateType.skywork,
        get_skywork_model_tokenizer,
        architectures=['SkyworkForCausalLM'],
        model_arch=ModelArch.llama,
    ))


def get_model_tokenizer_codellama(model_dir: str,
                                  model_info: ModelInfo,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=False)
    return get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.codefuse_codellama,
        [
            ModelGroup(
                [
                    Model('codefuse-ai/CodeFuse-CodeLlama-34B', 'codefuse-ai/CodeFuse-CodeLlama-34B'),
                ],
                tags=['coding', 'skip_test'],
            ),
        ],
        TemplateType.codefuse_codellama,
        get_model_tokenizer_codellama,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))


def get_model_tokenizer_yuan(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=True)
    addi_tokens = [
        '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
        '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>',
        '<empty_output>'
    ]
    tokenizer.add_tokens(addi_tokens, special_tokens=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.yuan2,
        [
            ModelGroup([
                Model('IEITYuan/Yuan2.0-2B-hf', 'IEITYuan/Yuan2-2B-hf'),
                Model('IEITYuan/Yuan2.0-51B-hf', 'IEITYuan/Yuan2-51B-hf'),
                Model('IEITYuan/Yuan2.0-102B-hf', 'IEITYuan/Yuan2-102B-hf'),
                Model('IEITYuan/Yuan2-2B-Janus-hf', 'IEITYuan/Yuan2-2B-Janus-hf'),
            ]),
            ModelGroup([
                Model('IEITYuan/Yuan2-M32-hf', 'IEITYuan/Yuan2-M32-hf'),
            ], tags=['moe', 'skip_test']),
        ],
        TemplateType.yuan,
        get_model_tokenizer_yuan,
        model_arch=ModelArch.yuan,
        architectures=['YuanForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.orion,
        [
            ModelGroup([
                Model('OrionStarAI/Orion-14B-Base', 'OrionStarAI/Orion-14B-Base'),
                Model('OrionStarAI/Orion-14B-Chat', 'OrionStarAI/Orion-14B-Chat'),
            ],
                       ignore_file_pattern=[r'.+\.gguf$']),
        ],
        TemplateType.orion,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['OrionForCausalLM'],
    ))


def get_model_tokenizer_idefics(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_dir)
    if 'automodel_class' not in kwargs:
        kwargs['automodel_class'] = AutoModelForVision2Seq
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.idefics3,
        [
            ModelGroup([
                Model('AI-ModelScope/Idefics3-8B-Llama3', 'HuggingFaceM4/Idefics3-8B-Llama3'),
            ],
                       tags=['multi-modal', 'vision'],
                       requires=['transformers>=4.45']),
        ],
        TemplateType.idefics3,
        get_model_tokenizer_idefics,
        model_arch=ModelArch.idefics3,
        architectures=['Idefics3ForConditionalGeneration'],
    ))


def get_model_tokenizer_mplug_owl2(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/X-PLUG/mPLUG-Owl')
    local_repo_path = os.path.join(local_repo_path, 'mPLUG-Owl2')
    sys.path.append(os.path.join(local_repo_path))

    # register
    # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py#L447
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vocab_size = kwargs.pop('vocab_size', None)
    if vocab_size is not None:
        model_config.vocab_size = vocab_size
    get_model_tokenizer_function = kwargs.pop('get_model_tokenizer_function')
    model, tokenizer = get_model_tokenizer_function(model_dir, model_info, model_kwargs, load_model, **kwargs)
    logger.info('Please ignore the unimported warning.')
    processor = CLIPImageProcessor.from_pretrained(model_dir)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.mplug2,
        [
            ModelGroup([
                Model('iic/mPLUG-Owl2', 'MAGAer13/mplug-owl2-llama2-7b'),
            ],
                       tags=['multi-modal', 'vision'],
                       requires=['transformers<4.35', 'icecream']),
        ],
        TemplateType.mplug_owl2,
        partial(get_model_tokenizer_mplug_owl2, get_model_tokenizer_function=get_model_tokenizer_with_flash_attn),
        model_arch=None,
    ))

register_model(
    ModelMeta(
        MLLMModelType.mplug2_1,
        [
            ModelGroup([
                Model('iic/mPLUG-Owl2.1', 'Mizukiluke/mplug_owl_2_1'),
            ],
                       tags=['multi-modal', 'vision'],
                       requires=['transformers<4.35', 'icecream']),
        ],
        TemplateType.mplug_owl2,
        partial(
            get_model_tokenizer_mplug_owl2, vocab_size=151851, get_model_tokenizer_function=get_model_tokenizer_qwen),
        model_arch=None,
    ))

register_model(
    ModelMeta(
        LLMModelType.wizardlm2_moe,
        [
            ModelGroup([
                Model('AI-ModelScope/WizardLM-2-8x22B', 'alpindale/WizardLM-2-8x22B'),
            ],
                       requires=['transformers>=4.36'], tags=['skip_test']),
        ],
        TemplateType.wizardlm2,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MixtralForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.wizardlm2_awq,
        [
            ModelGroup([
                Model('AI-ModelScope/WizardLM-2-7B-AWQ', 'MaziyarPanahi/WizardLM-2-7B-AWQ'),
            ],
                       requires=['transformers>=4.34'], tags=['skip_test']),
        ],
        TemplateType.wizardlm2_awq,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MistralForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.numina,
        [
            ModelGroup([
                Model('AI-ModelScope/NuminaMath-7B-TIR', 'AI-MO/NuminaMath-7B-TIR'),
            ], tags=['math']),
        ],
        TemplateType.numina_math,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_deepseek,
        [
            ModelGroup([
                Model('OpenBuddy/openbuddy-deepseek-67b-v15.2', 'OpenBuddy/openbuddy-deepseek-67b-v15.2'),
            ], tags=['skip_test']),
        ],
        TemplateType.openbuddy,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.sus,
        [
            ModelGroup([
                Model('SUSTC/SUS-Chat-34B', 'SUSTech/SUS-Chat-34B'),
            ], tags=['skip_test']),
        ],
        TemplateType.sus,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_zephyr,
        [
            ModelGroup(
                [
                    Model('OpenBuddy/openbuddy-zephyr-7b-v14.1', 'OpenBuddy/openbuddy-zephyr-7b-v14.1'),
                ],
                requires=['transformers>=4.34'],
            ),
        ],
        TemplateType.openbuddy,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MistralForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.zephyr,
        [
            ModelGroup([
                Model('modelscope/zephyr-7b-beta', 'HuggingFaceH4/zephyr-7b-beta'),
            ],
                       requires=['transformers>=4.34']),
        ],
        TemplateType.zephyr,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MistralForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.ziya2,
        [
            ModelGroup([
                Model('Fengshenbang/Ziya2-13B-Base', 'IDEA-CCNL/Ziya2-13B-Base'),
                Model('Fengshenbang/Ziya2-13B-Chat', 'IDEA-CCNL/Ziya2-13B-Chat'),
            ]),
        ],
        TemplateType.ziya,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_mixtral,
        [
            ModelGroup(
                [
                    Model('OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k', 'OpenBuddy/openbuddy-mixtral-7bx8-v18.1-32k'),
                ],
                requires=['transformers>=4.36'],
                tags=['moe', 'skip_test'],
            ),
        ],
        TemplateType.openbuddy,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MixtralForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_mistral,
        [
            ModelGroup([
                Model('OpenBuddy/openbuddy-mistral-7b-v17.1-32k', 'OpenBuddy/openbuddy-mistral-7b-v17.1-32k'),
            ],
                       requires=['transformers>=4.34']),
        ],
        TemplateType.openbuddy,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['MistralForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_llama,
        [
            ModelGroup([
                Model('OpenBuddy/openbuddy-llama-65b-v8-bf16', 'OpenBuddy/openbuddy-llama-65b-v8-bf16'),
            ], tags=['skip_test']),
        ],
        TemplateType.openbuddy,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.openbuddy_llama2,
        [
            ModelGroup([
                Model('OpenBuddy/openbuddy-llama2-13b-v8.1-fp16', 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16'),
                Model('OpenBuddy/openbuddy-llama2-70b-v10.1-bf16', 'OpenBuddy/openbuddy-llama2-70b-v10.1-bf16'),
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
            ]),
            ModelGroup([
                Model('OpenBuddy/openbuddy-llama3.1-8b-v22.1-131k', 'OpenBuddy/openbuddy-llama3.1-8b-v22.1-131k'),
            ],
                       requires=['transformers>=4.43']),
        ],
        TemplateType.openbuddy2,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.dbrx,
        [
            ModelGroup([
                Model('AI-ModelScope/dbrx-base', 'databricks/dbrx-base'),
                Model('AI-ModelScope/dbrx-instruct', 'databricks/dbrx-instruct'),
            ],
                       tags=['moe', 'skip_test'],
                       requires=['transformers>=4.36']),
        ],
        TemplateType.dbrx,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.dbrx,
        architectures=['DbrxForCausalLM'],
    ))


def get_model_tokenizer_ovis(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    if model is not None:
        model.generation_config.cache_implementation = None
        func_list = ['generate', 'forward', 'get_input_embeddings']
        use_submodel_func(model, 'llm', func_list)
        embedding = model.get_input_embeddings()
        embedding.register_forward_hook(patch_output_clone)
    try:
        # fix device_map
        from transformers.cache_utils import HybridCache

        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args,
                   **kwargs) -> Tuple[torch.Tensor]:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
            return self._update_origin(key_states, value_states, layer_idx, *args, **kwargs)

        if not hasattr(HybridCache, '_update_origin'):
            HybridCache._update_origin = HybridCache.update
            HybridCache.update = update
    except ImportError:
        pass
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.ovis1_6,
        [
            ModelGroup([
                Model('AIDC-AI/Ovis1.6-Gemma2-9B', 'AIDC-AI/Ovis1.6-Gemma2-9B'),
            ],
                       tags=['multi-modal', 'vision'],
                       requires=['transformers>=4.42']),
        ],
        TemplateType.ovis1_6,
        get_model_tokenizer_ovis,
        model_arch=ModelArch.ovis1_6,
        architectures=['Ovis'],
    ))

register_model(
    ModelMeta(
        LLMModelType.nenotron,
        [
            ModelGroup([
                Model('AI-ModelScope/Llama-3.1-Nemotron-70B-Instruct-HF', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'),
            ],
                       requires=['transformers>=4.43'],
                       ignore_file_pattern=[r'.+\.pth$'], tags=['skip_test']),
        ],
        TemplateType.llama3,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.reflection,
        [
            ModelGroup([
                Model('LLM-Research/Reflection-Llama-3.1-70B', 'mattshumer/Reflection-Llama-3.1-70B'),
            ],
                       requires=['transformers>=4.43'], tags=['skip_test']),
        ],
        TemplateType.reflection,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.atom,
        [
            ModelGroup([
                Model('FlagAlpha/Atom-7B', 'FlagAlpha/Atom-7B'),
                Model('FlagAlpha/Atom-7B-Chat', 'FlagAlpha/Atom-7B-Chat'),
            ]),
        ],
        TemplateType.atom,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.mengzi3,
        [
            ModelGroup([
                Model('langboat/Mengzi3-13B-Base', 'Langboat/Mengzi3-13B-Base'),
            ]),
        ],
        TemplateType.mengzi,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))


def get_model_tokenizer_got_ocr2(*args, **kwargs):
    if 'automodel_class' not in kwargs:
        kwargs['automodel_class'] = AutoModel
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.got_ocr2,
        [
            ModelGroup([
                Model('stepfun-ai/GOT-OCR2_0', 'stepfun-ai/GOT-OCR2_0'),
            ], tags=['multi-modal', 'audio']),
        ],
        TemplateType.got_ocr2,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.got_ocr2,
        architectures=['GOTQwenForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.bluelm,
        [
            ModelGroup([
                Model('vivo-ai/BlueLM-7B-Chat-32K', 'vivo-ai/BlueLM-7B-Chat-32K'),
                Model('vivo-ai/BlueLM-7B-Chat', 'vivo-ai/BlueLM-7B-Chat'),
                Model('vivo-ai/BlueLM-7B-Base-32K', 'vivo-ai/BlueLM-7B-Base-32K'),
                Model('vivo-ai/BlueLM-7B-Base', 'vivo-ai/BlueLM-7B-Base'),
            ]),
        ],
        TemplateType.bluelm,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.got_ocr2,
        architectures=['BlueLMForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.seggpt,
        [
            ModelGroup([
                Model('damo/nlp_seqgpt-560m', 'DAMO-NLP/SeqGPT-560M'),
            ], tags=['multi-modal', 'audio']),
        ],
        TemplateType.default,
        get_model_tokenizer_with_flash_attn,
        model_arch=None,
        architectures=['BloomForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.xverse,
        [
            ModelGroup([
                Model('xverse/XVERSE-7B', 'xverse/XVERSE-7B'),
                Model('xverse/XVERSE-7B-Chat', 'xverse/XVERSE-7B-Chat'),
                Model('xverse/XVERSE-13B', 'xverse/XVERSE-13B'),
                Model('xverse/XVERSE-13B-Chat', 'xverse/XVERSE-13B-Chat'),
                Model('xverse/XVERSE-65B', 'xverse/XVERSE-65B'),
                Model('xverse/XVERSE-65B-2', 'xverse/XVERSE-65B-2'),
                Model('xverse/XVERSE-65B-Chat', 'xverse/XVERSE-65B-Chat'),
                Model('xverse/XVERSE-13B-256K', 'xverse/XVERSE-13B-256K', ms_revision='v1.0.0'),
            ]),
        ],
        TemplateType.xverse,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['XverseForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.xverse_moe,
        [
            ModelGroup([
                Model('xverse/XVERSE-MoE-A4.2B', 'xverse/XVERSE-MoE-A4.2B'),
            ], tags=['moe']),
        ],
        TemplateType.xverse,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['XverseForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.c4ai,
        [
            ModelGroup([
                Model('AI-ModelScope/c4ai-command-r-v01', 'CohereForAI/c4ai-command-r-v01'),
                Model('AI-ModelScope/c4ai-command-r-plus', 'CohereForAI/c4ai-command-r-plus'),
            ],
                       requires=['transformers>=4.39']),
        ],
        TemplateType.c4ai,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['CohereForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.aya,
        [
            ModelGroup([
                Model('AI-ModelScope/aya-expanse-8b', 'CohereForAI/aya-expanse-8b'),
                Model('AI-ModelScope/aya-expanse-32b', 'CohereForAI/aya-expanse-32b'),
            ],
                       requires=['transformers>=4.44.0']),
        ],
        TemplateType.aya,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['CohereForCausalLM'],
    ))


def get_model_tokenizer_pixtral(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_dir)
    if 'automodel_class' not in kwargs:
        kwargs['automodel_class'] = LlavaForConditionalGeneration
    kwargs['tokenizer'] = processor.tokenizer
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.pixtral,
        [
            ModelGroup([
                Model('AI-ModelScope/pixtral-12b', 'mistral-community/pixtral-12b'),
            ],
                       tags=['multi-modal', 'vision'],
                       requires=['transformers>=4.45']),
        ],
        TemplateType.pixtral,
        get_model_tokenizer_pixtral,
        model_arch=ModelArch.llava,
        architectures=['LlavaForConditionalGeneration'],
    ))


def get_model_tokenizer_molmoe_1b(model_dir: str,
                                  model_info: ModelInfo,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor

    # fix bug for molmoe-1b
    def to_dict(self, *args, **kwargs):
        res = self._to_dict(*args, **kwargs)
        res['vision_backbone'] = self.vision_backbone.__dict__
        res.pop('to_dict')
        res.pop('_to_dict')
        return res

    model.config._to_dict = model.config.to_dict
    model.config.to_dict = MethodType(to_dict, model.config)
    from transformers import GenerationMixin
    model.generate = MethodType(GenerationMixin.generate, model)

    if model and hasattr(model, '_old_forward'):  # device_map
        device = model.lm_head.weight.device
        forward_origin = model._old_forward

        def _forward(*args, **kwargs):
            if 'append_last_valid_logits' in kwargs:
                kwargs['append_last_valid_logits'] = kwargs['append_last_valid_logits'].to(device)
            return forward_origin(*args, **kwargs)

        model._old_forward = _forward
        model.forward_origin = forward_origin

    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.molmoe_1b,
        [
            ModelGroup([
                Model('LLM-Research/MolmoE-1B-0924', 'allenai/MolmoE-1B-0924'),
            ],
                       tags=['multi-modal', 'vision'],
                       requires=['transformers>=4.45']),
        ],
        TemplateType.molmo,
        get_model_tokenizer_molmoe_1b,
        model_arch=ModelArch.molmo,
        support_gradient_checkpointing=False,
        torch_dtype=torch.float32,
        architectures=['MolmoForCausalLM'],
    ))


def get_model_tokenizer_molmo(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model_cls = get_class_from_dynamic_module('modeling_molmo.MolmoForCausalLM', model_dir)
    model_cls._no_split_modules = ['MolmoSequentialBlock']
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor
    if model:
        device = next(model.model.transformer.ff_out.parameters()).device
        forward_origin = model.model.forward

        def _forward(*args, **kwargs):
            if 'append_last_valid_logits' in kwargs:
                kwargs['append_last_valid_logits'] = kwargs['append_last_valid_logits'].to(device)
            return forward_origin(*args, **kwargs)

        model.model.forward = _forward
        model.model.forward_origin = forward_origin

    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.molmo,
        [
            ModelGroup([
                Model('LLM-Research/Molmo-7B-O-0924', 'allenai/Molmo-7B-O-0924'),
                Model('LLM-Research/Molmo-7B-D-0924', 'allenai/Molmo-7B-D-0924'),
                Model('LLM-Research/Molmo-72B-0924', 'allenai/Molmo-72B-0924'),
            ],
                       tags=['multi-modal', 'vision'],
                       requires=['transformers>=4.45']),
        ],
        TemplateType.molmo,
        get_model_tokenizer_molmo,
        model_arch=ModelArch.molmo,
        support_gradient_checkpointing=False,
        architectures=['MolmoForCausalLM'],
    ))

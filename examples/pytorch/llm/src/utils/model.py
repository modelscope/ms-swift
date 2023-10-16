# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from types import MethodType
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from modelscope import (AutoConfig, AutoModel, AutoModelForCausalLM,
                        AutoTokenizer, Model, read_config, snapshot_download)
from torch import dtype as Dtype
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils.versions import require_version

from swift import get_logger
from swift.utils import is_dist, is_local_master
from .preprocess import TemplateType

logger = get_logger()


def get_model_tokenizer_from_repo(model_dir: str,
                                  torch_dtype: Dtype,
                                  load_model: bool = True,
                                  model_config=None,
                                  tokenizer=None,
                                  automodel_class=AutoModelForCausalLM,
                                  **model_kwargs):
    """load from an independent repository"""
    if model_config is None:
        model_config = AutoConfig.from_pretrained(
            model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = automodel_class.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs)
    return model, tokenizer


def get_model_tokenizer_from_sdk(config_class: type,
                                 tokenizer_class: type,
                                 model_dir: str,
                                 torch_dtype: Dtype,
                                 load_model: bool = True,
                                 model_config=None,
                                 **model_kwargs):
    """load from ms library"""
    config = read_config(model_dir)
    logger.info(config)
    if model_config is None:
        model_config = config_class.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(model_config)
    tokenizer = tokenizer_class.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(
            model_dir,
            cfg_dict=config,
            config=model_config,
            torch_dtype=torch_dtype,
            **model_kwargs)
    return model, tokenizer


def get_model_tokenizer_baichuan_13b(model_dir: str,
                                     torch_dtype: Dtype,
                                     load_model: bool = True,
                                     **model_kwargs):
    # baichuan-13b does not implement the `get_input_embeddings` function
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     load_model,
                                                     **model_kwargs)

    if not hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings = MethodType(
            lambda self: self.model.embed_tokens, model)
    return model, tokenizer


def patch_baichuan2(self, hidden_states):
    # patch: baichuan2 lm_head
    if self.training:
        norm_weight = F.normalize(self.weight).to(self.weight.dtype)
    elif self.first_flag:
        self.first_flag = False
        self.weight.data = F.normalize(self.weight).to(self.weight.dtype)
        norm_weight = self.weight
    else:
        norm_weight = self.weight
    return F.linear(hidden_states, norm_weight)


def get_model_tokenizer_baichuan2_13b(model_dir: str,
                                      torch_dtype: Dtype,
                                      load_model: bool = True,
                                      **model_kwargs):
    # patch: baichuan2_13b configuration_baichuan.py bug
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    gradient_checkpointing = model_config.gradient_checkpointing
    if isinstance(gradient_checkpointing, (tuple, list)):
        model_config.gradient_checkpointing = gradient_checkpointing[0]
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     load_model, model_config,
                                                     **model_kwargs)
    if model is not None:
        model.lm_head.forward = MethodType(patch_baichuan2, model.lm_head)

    return model, tokenizer


def get_model_tokenizer_baichuan2_7b(model_dir: str,
                                     torch_dtype: Dtype,
                                     load_model: bool = True,
                                     **model_kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     load_model,
                                                     **model_kwargs)
    if model is not None:
        model.lm_head.forward = MethodType(patch_baichuan2, model.lm_head)

    return model, tokenizer


def get_model_tokenizer_chatglm2(model_dir: str,
                                 torch_dtype: Dtype,
                                 load_model: bool = True,
                                 **model_kwargs):
    if 'quantization_config' in model_kwargs:
        model_kwargs['quantization_config'].llm_int8_skip_modules = [
            'output_layer'
        ]
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, load_model,
                                         **model_kwargs)


def get_model_tokenizer_llama2(model_dir: str,
                               torch_dtype: Dtype,
                               load_model: bool = True,
                               **model_kwargs):
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, load_model,
                                         model_config, **model_kwargs)


def get_model_tokenizer_xverse(model_dir: str,
                               torch_dtype: Dtype,
                               load_model: bool = True,
                               **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return get_model_tokenizer_from_repo(
        model_dir,
        torch_dtype,
        load_model,
        tokenizer=tokenizer,
        **model_kwargs)


def get_model_tokenizer_polylm(model_dir: str,
                               torch_dtype: Dtype,
                               load_model: bool = True,
                               **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=False, legacy=True)
    return get_model_tokenizer_from_repo(
        model_dir,
        torch_dtype,
        load_model,
        tokenizer=tokenizer,
        **model_kwargs)


def get_model_tokenizer_qwen(model_dir: str,
                             torch_dtype: Dtype,
                             load_model: bool = True,
                             **kwargs):
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    dtype_mapping = {
        torch.float16: 'fp16',
        torch.bfloat16: 'bf16',
        torch.float32: 'fp32'
    }
    k_true = dtype_mapping[torch_dtype]
    for k in dtype_mapping.values():
        v = False
        if k == k_true:
            v = True
        setattr(model_config, k, v)

    use_flash_attn = kwargs.pop('use_flash_attn', 'auto')
    model_config.use_flash_attn = use_flash_attn
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     load_model, model_config,
                                                     **kwargs)
    try:
        # fix mp+ddp bug
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda(
        )
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    tokenizer.eos_token_id = tokenizer.eod_id
    return model, tokenizer


def get_model_tokenizer_qwen_vl(model_dir: str,
                                torch_dtype: Dtype,
                                load_model: bool = True,
                                **kwargs):
    if 'quantization_config' in kwargs:
        # https://github.com/pytorch/pytorch/issues/58969
        kwargs['quantization_config'].llm_int8_skip_modules = [
            'lm_head', 'attn_pool.attn'
        ]
    model, tokenizer = get_model_tokenizer_qwen(model_dir, torch_dtype,
                                                load_model, **kwargs)
    first_drop = model.transformer.drop
    if first_drop.p == 0.:
        # fix gradient_checkpointing bug
        _old_forward = first_drop.forward
        first_drop.forward = lambda *args, **kwargs: _old_forward(
            *args, **kwargs).clone()
    return model, tokenizer


class ModelType:
    # qwen
    qwen_7b = 'qwen-7b'
    qwen_7b_chat = 'qwen-7b-chat'
    qwen_14b = 'qwen-14b'
    qwen_14b_chat = 'qwen-14b-chat'
    # qwen-vl
    qwen_vl = 'qwen-vl'
    qwen_vl_chat = 'qwen-vl-chat'
    # baichuan
    baichuan_7b = 'baichuan-7b'
    baichuan_13b = 'baichuan-13b'
    baichuan_13b_chat = 'baichuan-13b-chat'
    baichuan2_7b = 'baichuan2-7b'
    baichuan2_7b_chat = 'baichuan2-7b-chat'
    baichuan2_13b = 'baichuan2-13b'
    baichuan2_13b_chat = 'baichuan2-13b-chat'
    # chatglm2
    chatglm2_6b = 'chatglm2-6b'
    chatglm2_6b_32k = 'chatglm2-6b-32k'
    # llama2
    llama2_7b = 'llama2-7b'
    llama2_13b = 'llama2-13b'
    llama2_70b = 'llama2-70b'
    llama2_7b_chat = 'llama2-7b-chat'
    llama2_13b_chat = 'llama2-13b-chat'
    llama2_70b_chat = 'llama2-70b-chat'
    # openbuddy
    openbuddy_llama2_13b_chat = 'openbuddy-llama2-13b-chat'
    openbuddy_llama2_65b_chat = 'openbuddy-llama-65b-chat'
    openbuddy_llama2_70b_chat = 'openbuddy-llama2-70b-chat'
    openbuddy_mistral_7b_chat = 'openbuddy-mistral-7b-chat'
    # internlm
    internlm_7b = 'internlm-7b'
    internlm_7b_chat = 'internlm-7b-chat'
    internlm_7b_chat_8k = 'internlm-7b-chat-8k'
    internlm_20b = 'internlm-20b'
    internlm_20b_chat = 'internlm-20b-chat'
    # xverse
    xverse_7b = 'xverse-7b'
    xverse_7b_chat = 'xverse-7b-chat'
    xverse_13b = 'xverse-13b'
    xverse_13b_chat = 'xverse-13b-chat'
    # mistral
    mistral_7b = 'mistral-7b'
    mistral_7b_chat = 'mistral-7b-chat'
    # ziya
    ziya2_13b = 'ziya2-13b'
    ziya2_13b_chat = 'ziya2-13b-chat'
    # other
    polylm_13b = 'polylm-13b'
    seqgpt_560m = 'seqgpt-560m'


class LoRATM(NamedTuple):
    # default lora target modules. qkv
    baichuan = ['W_pack']
    chatglm2 = ['query_key_value']
    llama2 = ['q_proj', 'k_proj', 'v_proj']
    qwen = ['c_attn']
    polylm = ['c_attn']
    bloom = ['query_key_value']
    internlm = ['q_proj', 'k_proj', 'v_proj']
    xverse = ['q_proj', 'k_proj', 'v_proj']
    mistral = ['q_proj', 'k_proj', 'v_proj']
    ziya = ['q_proj', 'k_proj', 'v_proj']


class AdapterTM(NamedTuple):
    # default adapter target modules.
    baichuan = ['mlp']
    chatglm2 = ['mlp']
    llama2 = ['mlp']
    qwen = ['mlp']
    polylm = ['mlp']


class ResTunerTM(NamedTuple):
    # default res-tuning config.
    baichuan = {
        'root_modules': r'.*layers.0$',
        'stem_modules': r'.*layers\.\d+$',
        'target_modules': r'.*model.norm',
        'target_modules_hook': 'input',
        'tuner_cfg': 'res_adapter',
    }
    chatglm2 = {
        'root_modules': r'.*layers.0$',
        'stem_modules': r'.*layers\.\d+$',
        'target_modules': r'.*final_layernorm',
        'target_modules_hook': 'input',
        'tuner_cfg': 'res_adapter',
    }
    llama2 = {
        'root_modules': r'.*layers.0$',
        'stem_modules': r'.*layers\.\d+$',
        'target_modules': r'.*model.norm',
        'target_modules_hook': 'input',
        'tuner_cfg': 'res_adapter',
    }
    qwen = {
        'root_modules': r'.*transformer.h.0$',
        'stem_modules': r'.*transformer.h\.\d+$',
        'target_modules': r'.*transformer.ln_f',
        'target_modules_hook': 'input',
        'tuner_cfg': 'res_adapter',
    }
    polylm = {
        'root_modules': r'.*transformer.h.0$',
        'stem_modules': r'.*transformer.h\.\d+$',
        'target_modules': r'.*transformer.ln_f',
        'target_modules_hook': 'input',
        'tuner_cfg': 'res_adapter',
    }


# Model Home: 'https://modelscope.cn/models/{model_id}/summary'
# model_id: model id or model dir
MODEL_MAPPING: Dict[str, Dict[str, Any]] = {
    # qwen
    ModelType.qwen_7b: {
        'model_id': 'qwen/Qwen-7B',
        'revision': 'v1.1.4',
        'get_function': get_model_tokenizer_qwen,
        'lora_TM': LoRATM.qwen,
    },
    ModelType.qwen_7b_chat: {
        'model_id': 'qwen/Qwen-7B-Chat',
        'revision': 'v1.1.4',
        'get_function': get_model_tokenizer_qwen,
        'template': TemplateType.chatml,
        'lora_TM': LoRATM.qwen,
    },
    ModelType.qwen_14b: {
        'model_id': 'qwen/Qwen-14B',
        'revision': 'v1.0.4',
        'get_function': get_model_tokenizer_qwen,
        'lora_TM': LoRATM.qwen,
    },
    ModelType.qwen_14b_chat: {
        'model_id': 'qwen/Qwen-14B-Chat',
        'revision': 'v1.0.4',
        'get_function': get_model_tokenizer_qwen,
        'template': TemplateType.chatml,
        'lora_TM': LoRATM.qwen,
    },
    # qwen-vl
    ModelType.qwen_vl: {
        'model_id': 'qwen/Qwen-VL',
        'revision': 'v1.0.3',
        'get_function': get_model_tokenizer_qwen_vl,
        'lora_TM': LoRATM.qwen,
    },
    ModelType.qwen_14b_chat: {
        'model_id': 'qwen/Qwen-VL-Chat',
        'revision': 'v1.1.0',
        'get_function': get_model_tokenizer_qwen_vl,
        'template': TemplateType.chatml,
        'lora_TM': LoRATM.qwen,
    },
    # baichuan
    ModelType.baichuan_7b: {
        'model_id': 'baichuan-inc/baichuan-7B',
        'revision': 'v1.0.7',
        'lora_TM': LoRATM.baichuan,
        'requires': ['transformers<4.34']
    },
    ModelType.baichuan_13b: {
        'model_id': 'baichuan-inc/Baichuan-13B-Base',
        'revision': 'v1.0.5',
        'get_function': get_model_tokenizer_baichuan_13b,
        'lora_TM': LoRATM.baichuan,
        'requires': ['transformers<4.34']
    },
    ModelType.baichuan_13b_chat: {
        'model_id': 'baichuan-inc/Baichuan-13B-Chat',
        'revision': 'v1.0.8',
        'template': TemplateType.baichuan,
        'lora_TM': LoRATM.baichuan,
        'requires': ['transformers<4.34']
    },
    ModelType.baichuan2_7b: {
        'model_id': 'baichuan-inc/Baichuan2-7B-Base',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_baichuan2_7b,
        'lora_TM': LoRATM.baichuan,
    },
    ModelType.baichuan2_7b_chat: {
        'model_id': 'baichuan-inc/Baichuan2-7B-Chat',
        'revision': 'v1.0.4',
        'template': TemplateType.baichuan,
        'get_function': get_model_tokenizer_baichuan2_7b,
        'lora_TM': LoRATM.baichuan,
    },
    ModelType.baichuan2_13b: {
        'model_id': 'baichuan-inc/Baichuan2-13B-Base',
        'revision': 'v1.0.3',
        'get_function': get_model_tokenizer_baichuan2_13b,
        'lora_TM': LoRATM.baichuan,
    },
    ModelType.baichuan2_13b_chat: {
        'model_id': 'baichuan-inc/Baichuan2-13B-Chat',
        'revision': 'v1.0.3',
        'template': TemplateType.baichuan,
        'get_function': get_model_tokenizer_baichuan2_13b,
        'lora_TM': LoRATM.baichuan,
    },
    # chatglm2
    ModelType.chatglm2_6b: {
        'model_id': 'ZhipuAI/chatglm2-6b',
        'revision': 'v1.0.12',
        'get_function': get_model_tokenizer_chatglm2,
        'template': TemplateType.chatglm2,
        'lora_TM': LoRATM.chatglm2,
    },
    ModelType.chatglm2_6b_32k: {
        'model_id': 'ZhipuAI/chatglm2-6b-32k',
        'revision': 'v1.0.2',
        'template': TemplateType.chatglm2,
        'lora_TM': LoRATM.chatglm2,
    },
    # llama
    ModelType.llama2_7b: {
        'model_id': 'modelscope/Llama-2-7b-ms',
        'revision': 'v1.0.2',
        'ignore_file_pattern': [r'.+\.bin$'],  # use safetensors
        'lora_TM': LoRATM.llama2,
    },
    ModelType.llama2_13b: {
        'model_id': 'modelscope/Llama-2-13b-ms',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_llama2,
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2,
    },
    ModelType.llama2_70b: {
        'model_id': 'modelscope/Llama-2-70b-ms',
        'revision': 'v1.0.0',
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2,
    },
    ModelType.llama2_7b_chat: {
        'model_id': 'modelscope/Llama-2-7b-chat-ms',
        'revision': 'v1.0.2',
        'template': TemplateType.llama,
        'ignore_file_pattern': [r'.+\.bin$'],  # use safetensors
        'lora_TM': LoRATM.llama2,
    },
    ModelType.llama2_13b_chat: {
        'model_id': 'modelscope/Llama-2-13b-chat-ms',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_llama2,
        'template': TemplateType.llama,
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2,
    },
    ModelType.llama2_70b_chat: {
        'model_id': 'modelscope/Llama-2-70b-chat-ms',
        'revision': 'v1.0.1',
        'get_function': get_model_tokenizer_llama2,
        'template': TemplateType.llama,
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2,
    },
    # openbuddy
    ModelType.openbuddy_llama2_13b_chat: {
        'model_id': 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
        'revision': 'v1.0.0',
        'template': TemplateType.openbuddy,
        'lora_TM': LoRATM.llama2,
    },
    ModelType.openbuddy_llama2_65b_chat: {
        'model_id': 'OpenBuddy/openbuddy-llama-65b-v8-bf16',
        'revision': 'v1.0.0',
        'template': TemplateType.openbuddy,
        'lora_TM': LoRATM.llama2,
    },
    ModelType.openbuddy_llama2_70b_chat: {
        'model_id': 'OpenBuddy/openbuddy-llama2-70b-v10.1-bf16',
        'revision': 'v1.0.0',
        'template': TemplateType.openbuddy,
        'lora_TM': LoRATM.llama2,
    },
    ModelType.openbuddy_mistral_7b_chat: {
        'model_id': 'OpenBuddy/openbuddy-mistral-7b-v13.1',
        'revision': 'v1.0.0',
        'template': TemplateType.openbuddy,
        'lora_TM': LoRATM.mistral,
        'requires': ['transformers>=4.34']
    },
    # internlm
    ModelType.internlm_7b: {
        'model_id': 'Shanghai_AI_Laboratory/internlm-7b',
        'revision': 'v1.0.1',
        'lora_TM': LoRATM.internlm,
    },
    ModelType.internlm_7b_chat: {
        'model_id': 'Shanghai_AI_Laboratory/internlm-chat-7b-v1_1',
        'revision': 'v1.0.1',
        'template': TemplateType.internlm,
        'lora_TM': LoRATM.internlm,
    },
    ModelType.internlm_7b_chat_8k: {
        'model_id': 'Shanghai_AI_Laboratory/internlm-chat-7b-8k',
        'revision': 'v1.0.1',
        'template': TemplateType.internlm,
        'lora_TM': LoRATM.internlm,
    },
    ModelType.internlm_20b: {
        'model_id': 'Shanghai_AI_Laboratory/internlm-20b',
        'revision': 'v1.0.1',
        'lora_TM': LoRATM.internlm,
    },
    ModelType.internlm_20b_chat: {
        'model_id': 'Shanghai_AI_Laboratory/internlm-chat-20b',
        'revision': 'v1.0.1',
        'template': TemplateType.internlm,
        'lora_TM': LoRATM.internlm,
    },
    # xverse
    ModelType.xverse_7b: {
        'model_id': 'xverse/XVERSE-7B',
        'revision': 'v1.0.0',
        'get_function': get_model_tokenizer_xverse,
        'lora_TM': LoRATM.xverse,
    },
    ModelType.xverse_7b_chat: {
        'model_id': 'xverse/XVERSE-7B-Chat',
        'revision': 'v1.0.0',
        'template': TemplateType.xverse,
        'get_function': get_model_tokenizer_xverse,
        'lora_TM': LoRATM.xverse,
    },
    ModelType.xverse_13b: {
        'model_id': 'xverse/XVERSE-13B',
        'revision': 'v1.0.0',
        'get_function': get_model_tokenizer_xverse,
        'lora_TM': LoRATM.xverse,
    },
    ModelType.xverse_13b_chat: {
        'model_id': 'xverse/XVERSE-13B-Chat',
        'revision': 'v1.0.0',
        'template': TemplateType.xverse,
        'get_function': get_model_tokenizer_xverse,
        'lora_TM': LoRATM.xverse,
    },
    # mistral
    ModelType.mistral_7b: {
        'model_id': 'AI-ModelScope/Mistral-7B-v0.1',
        'revision': 'v1.0.0',
        'lora_TM': LoRATM.mistral,
        'requires': ['transformers>=4.34']
    },
    ModelType.mistral_7b_chat: {
        'model_id': 'AI-ModelScope/Mistral-7B-Instruct-v0.1',
        'revision': 'v1.0.0',
        'template': TemplateType.llama,
        'lora_TM': LoRATM.mistral,
        'requires': ['transformers>=4.34']
    },
    # ziya
    ModelType.ziya2_13b: {
        'model_id': 'Fengshenbang/Ziya2-13B-Base',
        'revision': 'v1.0.0',
        'lora_TM': LoRATM.ziya,
    },
    ModelType.ziya2_13b_chat: {
        'model_id': 'Fengshenbang/Ziya2-13B-Chat',
        'revision': 'v1.0.0',
        'template': TemplateType.ziya,
        'lora_TM': LoRATM.ziya,
    },
    # other
    ModelType.polylm_13b: {
        'model_id': 'damo/nlp_polylm_13b_text_generation',
        'revision': 'v1.0.3',
        'get_function': get_model_tokenizer_polylm,
        'lora_TM': LoRATM.polylm,
    },
    ModelType.seqgpt_560m: {
        'model_id': 'damo/nlp_seqgpt-560m',
        'revision': 'v1.0.1',
        'template': TemplateType.default_generation,
        'lora_TM': LoRATM.bloom,
    },
}


def get_model_tokenizer(
        model_type: str,
        torch_dtype: Optional[Dtype] = None,
        load_model: bool = True,
        **kwargs) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizer]:
    data = MODEL_MAPPING.get(model_type)
    requires = data.get('requires', [])
    for require in requires:
        require_version(require)
    if data is None:
        raise ValueError(f'model_type: {model_type}')

    model_id = data['model_id']
    get_function = data.get('get_function', get_model_tokenizer_from_repo)
    ignore_file_pattern = data.get('ignore_file_pattern', [])
    if torch_dtype is None:
        torch_dtype = data.get('torch_dtype', torch.float16)
    if 'device_map' not in kwargs:
        kwargs['device_map'] = 'auto'

    model_dir = kwargs.pop('model_dir', None)
    if model_dir is None:
        if is_dist() and not is_local_master():
            dist.barrier()
        model_dir = model_id
        if not os.path.exists(model_id):
            revision = data.get('revision', 'master')
            model_dir = snapshot_download(
                model_id, revision, ignore_file_pattern=ignore_file_pattern)
        if is_dist() and is_local_master():
            dist.barrier()

    model, tokenizer = get_function(model_dir, torch_dtype, load_model,
                                    **kwargs)
    assert tokenizer.eos_token is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

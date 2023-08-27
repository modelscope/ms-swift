import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from types import MethodType
from typing import NamedTuple, Optional

import torch
from modelscope import (AutoConfig, AutoModel, AutoModelForCausalLM,
                        AutoTokenizer, Model, read_config, snapshot_download)
from torch import dtype as Dtype

from swift import get_logger
from .utils import broadcast_string, is_dist, is_master

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


def get_model_tokenizer_baichuan13b(model_dir: str,
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


def get_model_tokenizer_chatglm2(model_dir: str,
                                 torch_dtype: Dtype,
                                 load_model: bool = True,
                                 **model_kwargs):
    if 'quantization_config' in model_kwargs:
        model_kwargs['quantization_config'].llm_int8_skip_modules = [
            'output_layer'
        ]
    return get_model_tokenizer_from_repo(
        model_dir,
        torch_dtype,
        load_model,
        automodel_class=AutoModel,
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


def get_model_tokenizer_polylm(model_dir: str,
                               torch_dtype: Dtype,
                               load_model: bool = True,
                               **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=False)
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
    mapper = {
        torch.float16: 'fp16',
        torch.bfloat16: 'bf16',
        torch.float32: 'fp32'
    }
    k_true = mapper[torch_dtype]
    for k in mapper.values():
        v = False
        if k == k_true:
            v = True
        setattr(model_config, k, v)

    use_flash_attn = kwargs.pop('use_flash_attn', 'auto')
    model_config.use_flash_attn = use_flash_attn
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     load_model, model_config,
                                                     **kwargs)
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
    return get_model_tokenizer_qwen(model_dir, torch_dtype, load_model,
                                    **kwargs)


class LoRATM(NamedTuple):
    # default lora target modules. qkv
    baichuan = ['W_pack']
    chatglm2 = ['query_key_value']
    llama2 = ['q_proj', 'k_proj', 'v_proj']
    qwen = ['c_attn']
    polylm = ['c_attn']


# Model Home: 'https://modelscope.cn/models/{model_id}/summary'
# keys: 'model_id', 'revision', 'get_function', 'template',
#   'ignore_file_pattern', 'lora_TM'
MODEL_MAPPING = {
    'qwen-7b': {
        'model_id': 'qwen/Qwen-7B',  # model id or model dir
        'revision': 'v1.0.5',
        'get_function': get_model_tokenizer_qwen,
        'template': 'chatml',
        'lora_TM': LoRATM.qwen,
    },
    'qwen-7b-chat': {
        'model_id': 'qwen/Qwen-7B-Chat',
        'revision': 'v1.0.6',
        'get_function': get_model_tokenizer_qwen,
        'template': 'chatml',
        'lora_TM': LoRATM.qwen,
    },
    'qwen-vl': {
        'model_id': 'qwen/Qwen-VL',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_qwen_vl,
        'template': 'chatml',
        'lora_TM': LoRATM.qwen,
    },
    'qwen-vl-chat': {
        'model_id': 'qwen/Qwen-VL-Chat',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_qwen_vl,
        'template': 'chatml',
        'lora_TM': LoRATM.qwen,
    },
    'baichuan-7b': {
        'model_id': 'baichuan-inc/baichuan-7B',
        'revision': 'v1.0.7',
        'template': 'baichuan',
        'lora_TM': LoRATM.baichuan,
    },
    'baichuan-13b': {
        'model_id': 'baichuan-inc/Baichuan-13B-Base',
        'revision': 'v1.0.5',
        'get_function': get_model_tokenizer_baichuan13b,
        'template': 'baichuan',
        'lora_TM': LoRATM.baichuan,
    },
    'baichuan-13b-chat': {
        'model_id': 'baichuan-inc/Baichuan-13B-Chat',
        'revision': 'v1.0.8',
        'template': 'baichuan',
        'lora_TM': LoRATM.baichuan,
    },
    'chatglm2-6b': {
        'model_id': 'ZhipuAI/chatglm2-6b',
        'revision': 'v1.0.8',
        'get_function': get_model_tokenizer_chatglm2,
        'template': 'chatglm2',
        'lora_TM': LoRATM.chatglm2,
    },
    'chatglm2-6b-32k': {
        'model_id': 'ZhipuAI/chatglm2-6b-32k',
        'revision': 'v1.0.0',
        'template': 'chatglm2',
        'lora_TM': LoRATM.chatglm2,
    },
    'llama2-7b': {
        'model_id': 'modelscope/Llama-2-7b-ms',
        'revision': 'v1.0.2',
        'template': 'llama',
        'ignore_file_pattern': [r'.+\.bin$'],  # use safetensors
        'lora_TM': LoRATM.llama2,
    },
    'llama2-13b': {
        'model_id': 'modelscope/Llama-2-13b-ms',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_llama2,
        'template': 'llama',
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2,
    },
    'llama2-70b': {
        'model_id': 'modelscope/Llama-2-70b-ms',
        'revision': 'v1.0.0',
        'template': 'llama',
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2,
    },
    'llama2-7b-chat': {
        'model_id': 'modelscope/Llama-2-7b-chat-ms',
        'revision': 'v1.0.2',
        'template': 'llama',
        'ignore_file_pattern': [r'.+\.bin$'],  # use safetensors
        'lora_TM': LoRATM.llama2,
    },
    'llama2-13b-chat': {
        'model_id': 'modelscope/Llama-2-13b-chat-ms',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_llama2,
        'template': 'llama',
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2,
    },
    'llama2-70b-chat': {
        'model_id': 'modelscope/Llama-2-70b-chat-ms',
        'revision': 'v1.0.1',
        'get_function': get_model_tokenizer_llama2,
        'template': 'llama',
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2,
    },
    'openbuddy-llama2-13b': {
        'model_id': 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
        'revision': 'v1.0.0',
        'template': 'openbuddy_llama',
        'lora_TM': LoRATM.llama2,
    },
    'openbuddy-llama-65b': {
        'model_id': 'OpenBuddy/openbuddy-llama-65b-v8-bf16',
        'revision': 'v1.0.0',
        'template': 'openbuddy_llama',
        'lora_TM': LoRATM.llama2,
    },
    'polylm-13b': {
        'model_id': 'damo/nlp_polylm_13b_text_generation',
        'revision': 'v1.0.3',
        'get_function': get_model_tokenizer_polylm,
        'lora_TM': LoRATM.polylm,
    },
}


def get_model_tokenizer(model_type: str,
                        torch_dtype: Optional[Dtype] = None,
                        load_model: bool = True,
                        **kwargs):
    data = MODEL_MAPPING.get(model_type)
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
        if is_master():
            model_dir = model_id
            if not os.path.exists(model_id):
                revision = data.get('revision', 'master')
                model_dir = snapshot_download(
                    model_id,
                    revision,
                    ignore_file_pattern=ignore_file_pattern)
        if is_dist():
            model_dir = broadcast_string(model_dir)

    model, tokenizer = get_function(model_dir, torch_dtype, load_model,
                                    **kwargs)
    assert tokenizer.eos_token is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

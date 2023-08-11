import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from types import MethodType
from typing import Any, Dict, NamedTuple, Optional

import torch
from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Model,
                        read_config, snapshot_download)
from modelscope.models.nlp.chatglm2 import ChatGLM2Config, ChatGLM2Tokenizer
from modelscope.models.nlp.llama2 import Llama2Config, Llama2Tokenizer
from torch import dtype as Dtype

from swift import get_logger

logger = get_logger()


def _add_special_token(tokenizer, special_token_mapper: Dict[str,
                                                             Any]) -> None:
    for k, v in special_token_mapper.items():
        setattr(tokenizer, k, v)
    assert tokenizer.eos_token is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def get_model_tokenizer_from_repo(model_dir: str,
                                  torch_dtype: Dtype,
                                  load_model: bool = True,
                                  model_config=None,
                                  tokenizer=None,
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
        model = AutoModelForCausalLM.from_pretrained(
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
    return get_model_tokenizer_from_sdk(ChatGLM2Config, ChatGLM2Tokenizer,
                                        model_dir, torch_dtype, load_model,
                                        **model_kwargs)


def get_model_tokenizer_llama2(model_dir: str,
                               torch_dtype: Dtype,
                               load_model: bool = True,
                               **model_kwargs):
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    return get_model_tokenizer_from_sdk(Llama2Config, Llama2Tokenizer,
                                        model_dir, torch_dtype, load_model,
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
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, load_model,
                                         model_config, **kwargs)


class LoRATM(NamedTuple):
    # default lora target modules
    baichuan = ['W_pack']
    chatglm2 = ['query_key_value']
    llama2 = ['q_proj', 'k_proj', 'v_proj']
    qwen = ['c_attn']
    polylm = ['c_attn']


# Model Home: 'https://modelscope.cn/models/{model_id}/summary'
# keys: 'model_id', 'revision', 'get_function',
#   'ignore_file_pattern', 'special_token_mapper', 'lora_TM'
MODEL_MAPPING = {
    'qwen-7b': {
        'model_id': 'qwen/Qwen-7B',
        'revision': 'v.1.0.4',
        'get_function': get_model_tokenizer_qwen,
        'lora_TM': LoRATM.qwen,
        'special_token_mapper': {
            'eos_token': '<|endoftext|>'
        }
    },
    'baichuan-7b': {
        'model_id': 'baichuan-inc/baichuan-7B',  # model id or model dir
        'revision': 'v1.0.7',
        'lora_TM': LoRATM.baichuan
    },
    'baichuan-13b': {
        'model_id': 'baichuan-inc/Baichuan-13B-Base',
        'revision': 'v1.0.5',
        'get_function': get_model_tokenizer_baichuan13b,
        'lora_TM': LoRATM.baichuan
    },
    'chatglm2-6b': {
        'model_id': 'ZhipuAI/chatglm2-6b',
        'revision': 'v1.0.7',
        'get_function': get_model_tokenizer_chatglm2,
        'lora_TM': LoRATM.chatglm2
    },
    'llama2-7b': {
        'model_id': 'modelscope/Llama-2-7b-ms',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_llama2,
        'ignore_file_pattern': [r'.+\.bin$'],  # use safetensors
        'lora_TM': LoRATM.llama2
    },
    'llama2-13b': {
        'model_id': 'modelscope/Llama-2-13b-ms',
        'revision': 'v1.0.2',
        'get_function': get_model_tokenizer_llama2,
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2
    },
    'llama2-70b': {
        'model_id': 'modelscope/Llama-2-70b-ms',
        'revision': 'v1.0.0',
        'get_function': get_model_tokenizer_llama2,
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2
    },
    'openbuddy-llama2-13b': {
        'model_id': 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
        'revision': 'v1.0.0',
        'lora_TM': LoRATM.llama2,
    },
    'openbuddy-llama-65b': {
        'model_id': 'OpenBuddy/openbuddy-llama-65b-v8-bf16',
        'revision': 'v1.0.0',
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
    special_token_mapper = data.get('special_token_mapper', {})
    if torch_dtype is None:
        torch_dtype = data.get('torch_dtype', torch.float16)
    if 'device_map' not in kwargs:
        kwargs['device_map'] = 'auto'

    model_dir = kwargs.pop('model_dir', None)
    if model_dir is None:
        model_dir = model_id
        if not os.path.exists(model_id):
            revision = data.get('revision', 'master')
            model_dir = snapshot_download(
                model_id, revision, ignore_file_pattern=ignore_file_pattern)

    model, tokenizer = get_function(model_dir, torch_dtype, load_model,
                                    **kwargs)
    _add_special_token(tokenizer, special_token_mapper)
    return model, tokenizer

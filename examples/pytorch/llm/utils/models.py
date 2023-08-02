import os
from typing import Any, Dict, NamedTuple, Optional

import torch
from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Model,
                        get_logger, read_config, snapshot_download)
from modelscope.models.nlp.chatglm2 import ChatGLM2Config, ChatGLM2Tokenizer
from torch import dtype as Dtype

logger = get_logger()


def _add_special_token(tokenizer, special_token_mapper: Dict[str,
                                                             Any]) -> None:
    for k, v in special_token_mapper:
        setattr(tokenizer, k, v)
    assert tokenizer.eos_token is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def get_model_tokenizer_default(model_dir: str,
                                torch_dtype: Dtype,
                                load_model: bool = True):
    """load from an independent repository"""
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            device_map='auto',
            torch_dtype=torch_dtype,
            trust_remote_code=True)
    return model, tokenizer


def get_model_tokenizer_chatglm2(model_dir: str,
                                 torch_dtype: Dtype,
                                 load_model: bool = True):
    """load from ms library"""
    config = read_config(model_dir)
    logger.info(config)
    model_config = ChatGLM2Config.from_pretrained(model_dir)
    model_config.torch_dtype = torch_dtype
    logger.info(model_config)
    tokenizer = ChatGLM2Tokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(
            model_dir,
            cfg_dict=config,
            config=model_config,
            device_map='auto',
            torch_dtype=torch_dtype)
    return model, tokenizer


class LoRATM(NamedTuple):
    # default lora target modules
    baichuan = ['W_pack']
    chatglm2 = ['query_key_value']
    llama2 = ['q_proj', 'k_proj', 'v_proj']


# Reference: 'https://modelscope.cn/models/{model_id}/summary'
# keys: 'model_id', 'revision', 'torch_dtype', 'get_function',
#   'ignore_file_pattern', 'special_token_mapper', 'lora_TM'
MODEL_MAPPING = {
    'baichuan-7b': {
        'model_id': 'baichuan-inc/baichuan-7B',  # model id or model dir
        'revision': 'v1.0.7',
        'lora_TM': LoRATM.baichuan
    },
    'baichuan-13b': {
        'model_id': 'baichuan-inc/Baichuan-13B-Base',
        'revision': 'v1.0.3',
        'torch_dtype': torch.bfloat16,
        'lora_TM': LoRATM.baichuan
    },
    'chatglm2-6b': {
        'model_id': 'ZhipuAI/chatglm2-6b',
        'revision': 'v1.0.6',
        'get_function': get_model_tokenizer_chatglm2,
        'lora_TM': LoRATM.chatglm2
    },
    'llama2-7b': {
        'model_id': 'modelscope/Llama-2-7b-ms',
        'revision': 'v1.0.2',
        'ignore_file_pattern': [r'.+\.bin$'],  # use safetensors
        'lora_TM': LoRATM.llama2
    },
    'llama2-13b': {
        'model_id': 'modelscope/Llama-2-13b-ms',
        'revision': 'v1.0.2',
        'ignore_file_pattern': [r'.+\.bin$'],
        'lora_TM': LoRATM.llama2
    },
    'openbuddy-llama2-13b': {
        'model_id': 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
        'lora_TM': LoRATM.llama2,
        'revision': 'v1.0.0',
    }
}


def get_model_tokenizer(model_type: str,
                        torch_dtype: Optional[Dtype] = None,
                        load_model: bool = True):
    data = MODEL_MAPPING.get(model_type)
    if data is None:
        raise ValueError(f'model_type: {model_type}')

    model_id = data['model_id']
    get_function = data.get('get_function', get_model_tokenizer_default)
    ignore_file_pattern = data.get('ignore_file_pattern', [])
    special_token_mapper = data.get('special_token_mapper', {})
    if torch_dtype is None:
        torch_dtype = data.get('torch_dtype', torch.float16)

    model_dir = model_id
    if not os.path.exists(model_id):
        revision = data.get('revision', 'master')
        model_dir = snapshot_download(
            model_id, revision, ignore_file_pattern=ignore_file_pattern)

    model, tokenizer = get_function(model_dir, torch_dtype, load_model)
    _add_special_token(tokenizer, special_token_mapper)
    return model, tokenizer

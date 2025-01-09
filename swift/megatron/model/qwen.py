# Copyright (c) Alibaba, Inc. and its affiliates.

from swift.llm import ModelInfo, ModelGroup, Model
from .register import register_megatron_model, MegatronModelMeta
from .utils import get_model_provider
from .constant import MegatronModelType
from .config import load_config

def load_qwen_config(model_info: ModelInfo):
    args_config = load_config(model_info)
    args_config['swiglu'] = True
    return args_config

def convert_megatron2hf():
    pass

def convert_hf2megatron():
    pass


register_megatron_model(MegatronModelMeta(
MegatronModelType.qwen,[
        ModelGroup([
            Model('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'),
            Model('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct'),
            Model('Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-3B-Instruct'),
            Model('Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'),
            Model('Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'),
            Model('Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-32B-Instruct'),
            Model('Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2.5-72B-Instruct'),
        ]),
    ],
    convert_megatron2hf,
    convert_hf2megatron,
    get_model_provider,
    load_qwen_config
))

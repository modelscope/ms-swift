# Copyright (c) Alibaba, Inc. and its affiliates.

import importlib
from typing import Any, Dict

from megatron.training import get_args

from swift.llm import Model, ModelGroup, ModelInfo
from .config import load_config
from .constant import MegatronModelType
from .register import MegatronModelMeta, register_megatron_model
from .utils import get_model_provider


def load_qwen_config(model_info: ModelInfo) -> Dict[str, Any]:
    args_config = load_config(model_info)
    args_config['swiglu'] = True
    return args_config


def convert_megatron2hf(hf_model, mg_model):
    pass


def convert_hf2megatron(hf_model, mg_model):
    args = get_args()
    mg_model.to(args.torch_dtype)
    convert_module.save_mgmodel(mg_model, args)


def get_qwen_model_provider():
    module_prefix = 'megatron_patch.model.qwen2'
    gpt_model_cls = importlib.import_module(f'{module_prefix}.model').GPTModel
    transformer_config_cls = getattr(
        importlib.import_module(f'{module_prefix}.transformer_config'), 'Qwen2TransformerConfig')
    layer_spec_module = importlib.import_module(f'{module_prefix}.layer_specs')
    model_provider = get_model_provider(gpt_model_cls, transformer_config_cls, layer_spec_module)
    return model_provider


register_megatron_model(
    MegatronModelMeta(MegatronModelType.qwen, [
        ModelGroup([
            Model('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'),
            Model('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct'),
            Model('Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-3B-Instruct'),
            Model('Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'),
            Model('Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'),
            Model('Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-32B-Instruct'),
            Model('Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2.5-72B-Instruct'),
        ]),
    ], convert_megatron2hf, convert_hf2megatron, get_model_provider, load_qwen_config))

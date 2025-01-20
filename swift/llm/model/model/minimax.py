# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict

import json
import torch.cuda
from transformers import AutoConfig, AutoProcessor

from swift.llm import TemplateType
from swift.utils import get_dist_setting, get_logger
from ..constant import LLMModelType, MLLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo

logger = get_logger()


def get_model_tokenizer_minimax_vl(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    logger.warn('NOTE: minimax-vl-01 model does not support training.')
    n_gpu = torch.cuda.device_count()
    _, local_rank, _, local_world_size = get_dist_setting()
    if local_rank == -1:
        local_rank = 0
    device_ids = list(range(local_rank, n_gpu, local_world_size))
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['model_config'] = config
    if kwargs.get('attn_impl') == 'flash_attn':
        config.attention_type = 1
    else:
        config.attention_type = 0
    if 'quantization_config' in kwargs:
        quantization_config = kwargs['quantization_config']
        from transformers import QuantoConfig
        if isinstance(quantization_config, QuantoConfig):
            quantization_config.modules_to_not_convert = (
                [
                    'vision_tower',
                    'image_newline',
                    'multi_modal_projector',
                    'lm_head',
                    'embed_tokens',
                ] + [f'model.layers.{i}.coefficient' for i in range(config.text_config.num_hidden_layers)]
                + [f'model.layers.{i}.block_sparse_moe.gate' for i in range(config.text_config.num_hidden_layers)])

    if len(device_ids) > 1:
        model_safetensors_index_path = os.path.join(model_dir, 'model.safetensors.index.json')
        with open(model_safetensors_index_path, 'r') as f:
            model_safetensors_index = json.load(f)
        weight_map = model_safetensors_index['weight_map']
        vision_map = {}
        for key, value in weight_map.items():
            if 'vision_tower' in key or 'image_newline' in key or 'multi_modal_projector' in key:
                new_key = key.replace('.weight', '').replace('.bias', '')
                if new_key not in vision_map:
                    vision_map[new_key] = value

        device_map = {
            'language_model.model.embed_tokens': f'cuda:{device_ids[0]}',
            'language_model.model.norm': f'cuda:{device_ids[len(device_ids) - 1]}',
            'language_model.lm_head': f'cuda:{device_ids[len(device_ids) - 1]}'
        }
        for key, value in vision_map.items():
            device_map[key] = f'cuda:{device_ids[0]}'
        device_map['vision_tower.vision_model.post_layernorm'] = f'cuda:{device_ids[0]}'
        layers_per_device = config.text_config.num_hidden_layers // len(device_ids)
        for i in range(len(device_ids)):
            for j in range(layers_per_device):
                device_map[f'language_model.model.layers.{i * layers_per_device + j}'] = f'cuda:{device_ids[i]}'
        model_kwargs['device_map'] = device_map

    with open(os.path.join(model_dir, '__init__.py'), 'w') as f:
        f.write('')
    sys.path.insert(0, model_dir)
    from processing_minimax_vl_01 import MiniMaxVL01ProcessorKwargs, get_hw_multiple_of, get_num_token
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    processor.MiniMaxVL01ProcessorKwargs = MiniMaxVL01ProcessorKwargs
    processor.get_hw_multiple_of = get_hw_multiple_of
    processor.get_num_token = get_num_token
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    processor.tokenizer = tokenizer
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.minimax_vl, [
            ModelGroup([
                Model('MiniMaxAI/MiniMax-VL-01', 'MiniMaxAI/MiniMax-VL-01'),
            ]),
        ],
        TemplateType.minimax_vl,
        get_model_tokenizer_minimax_vl,
        architectures=['MiniMaxVL01ForConditionalGeneration']))


def get_model_tokenizer_minimax_text(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    logger.warn('NOTE: minimax-text-01 model does not support training.')
    n_gpu = torch.cuda.device_count()
    _, local_rank, _, local_world_size = get_dist_setting()
    if local_rank == -1:
        local_rank = 0
    device_ids = list(range(local_rank, n_gpu, local_world_size))
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['model_config'] = config
    if kwargs.get('attn_impl') == 'flash_attn':
        config.attention_type = 1
    else:
        config.attention_type = 0
    if 'quantization_config' in kwargs:
        quantization_config = kwargs['quantization_config']
        from transformers import QuantoConfig
        if isinstance(quantization_config, QuantoConfig):
            quantization_config.modules_to_not_convert = (
                [
                    'lm_head',
                    'embed_tokens',
                ] + [f'model.layers.{i}.coefficient' for i in range(config.num_hidden_layers)]
                + [f'model.layers.{i}.block_sparse_moe.gate' for i in range(config.num_hidden_layers)])

    if len(device_ids) > 1:
        layers_per_device = config.num_hidden_layers // len(device_ids)
        # set device map
        device_map = {
            'model.embed_tokens': 'cuda:0',
            'model.norm': f'cuda:{len(device_ids) - 1}',
            'lm_head': f'cuda:{len(device_ids) - 1}'
        }
        for i in range(len(device_ids)):
            for j in range(layers_per_device):
                device_map[f'model.layers.{i * layers_per_device + j}'] = f'cuda:{i}'
        model_kwargs['device_map'] = device_map

    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.minimax, [
            ModelGroup([
                Model('MiniMaxAI/MiniMax-Text-01', 'MiniMaxAI/MiniMax-Text-01'),
            ]),
        ],
        TemplateType.minimax,
        get_model_tokenizer_minimax_text,
        architectures=['MiniMaxText01ForCausalLM']))

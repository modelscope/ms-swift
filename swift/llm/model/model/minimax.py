# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import json
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from swift.utils import get_device, get_device_count, get_dist_setting, get_logger
from ..constant import LLMModelType, MLLMModelType
from ..patcher import patch_ignore_check_imports
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo

logger = get_logger()


def get_model_tokenizer_minimax_vl(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    logger.warn('NOTE: minimax-vl-01 model does not support training.')
    n_gpu = get_device_count()
    _, local_rank, _, local_world_size = get_dist_setting()
    device_ids = list(range(max(local_rank, 0), n_gpu, local_world_size))
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['model_config'] = config
    if kwargs.get('attn_impl') == 'flash_attn':
        config.attn_type_list = [1] * len(config.attn_type_list)
    else:
        config.attn_type_list = [0] * len(config.attn_type_list)
    if 'quantization_config' in model_kwargs:
        quantization_config = model_kwargs['quantization_config']
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
            'language_model.model.embed_tokens': get_device(device_ids[0]),
            'language_model.model.norm': get_device(device_ids[len(device_ids) - 1]),
            'language_model.lm_head': get_device(device_ids[len(device_ids) - 1])
        }
        for key, value in vision_map.items():
            device_map[key] = get_device(device_ids[0])
        device_map['vision_tower.vision_model.post_layernorm'] = get_device(device_ids[0])
        layers_per_device = config.text_config.num_hidden_layers // len(device_ids)
        for i in range(len(device_ids)):
            for j in range(layers_per_device):
                device_map[f'language_model.model.layers.{i * layers_per_device + j}'] = get_device(device_ids[i])
        model_kwargs['device_map'] = device_map

    MiniMaxVL01ProcessorKwargs = get_class_from_dynamic_module('processing_minimax_vl_01.MiniMaxVL01ProcessorKwargs',
                                                               model_dir)
    get_hw_multiple_of = get_class_from_dynamic_module('processing_minimax_vl_01.get_hw_multiple_of', model_dir)
    get_num_token = get_class_from_dynamic_module('processing_minimax_vl_01.get_num_token', model_dir)

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    processor.MiniMaxVL01ProcessorKwargs = MiniMaxVL01ProcessorKwargs
    processor.get_hw_multiple_of = get_hw_multiple_of
    processor.get_num_token = get_num_token
    with patch_ignore_check_imports():
        model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model,
                                                               **kwargs)
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
        architectures=['MiniMaxVL01ForConditionalGeneration'],
        tags=['vision']))


def get_model_tokenizer_minimax_text(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    logger.warn('NOTE: minimax-text-01 model does not support training.')
    n_gpu = get_device_count()
    _, local_rank, _, local_world_size = get_dist_setting()
    device_ids = list(range(max(local_rank, 0), n_gpu, local_world_size))
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['model_config'] = config
    if kwargs.get('attn_impl') == 'flash_attn':
        config.attn_type_list = [1] * len(config.attn_type_list)
    else:
        config.attn_type_list = [0] * len(config.attn_type_list)
    if 'quantization_config' in model_kwargs:
        quantization_config = model_kwargs['quantization_config']
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
            'model.embed_tokens': get_device(0),
            'model.norm': get_device(len(device_ids) - 1),
            'lm_head': get_device(len(device_ids) - 1)
        }
        for i in range(len(device_ids)):
            for j in range(layers_per_device):
                device_map[f'model.layers.{i * layers_per_device + j}'] = get_device(i)
        model_kwargs['device_map'] = device_map
    with patch_ignore_check_imports():
        model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model,
                                                               **kwargs)
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

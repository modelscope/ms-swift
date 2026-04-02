# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import fields
from mcore_bridge import ModelConfig
from mcore_bridge import get_mcore_model as _get_mcore_model
from mcore_bridge import hf_to_mcore_config
from transformers.utils import is_torch_npu_available

from swift.utils import get_logger

logger = get_logger()


def _check_attention_backend(args, config):
    """Validate attention backend compatibility with configuration."""
    attention_backend = config.attention_backend.name
    if attention_backend == 'flash' and config.softmax_type == 'learnable':
        raise ValueError(f'Attention backend "{attention_backend}" does not support learnable softmax_type.')


def _check_padding_free(args, config):
    """Validate and adjust padding_free setting based on configuration constraints."""
    if not args.padding_free:
        return

    attention_backend = config.attention_backend.name
    message = None

    if config.experimental_attention_variant == 'dsa':
        message = 'DSA is not supported in padding-free mode'
    elif attention_backend == 'unfused':
        message = f'Attention backend "{attention_backend}" is not supported in padding-free mode'

    if message:
        logger.warning(f'{message}. Setting args.padding_free to False.')
        args.padding_free = False


def get_mcore_model_config(args, hf_config):
    kwargs = hf_to_mcore_config(hf_config)
    kwargs['mcore_model_type'] = args.megatron_model_meta.model_type
    kwargs['hf_config'] = hf_config
    for f in fields(ModelConfig):
        key, value = f.name, getattr(args, f.name, None)
        if value is None or isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        kwargs[key] = value

    if args.task_type == 'seq_cls':
        args.problem_type = args.problem_type or getattr(hf_config, 'problem_type', None)
        logger.info(f'args.problem_type: {args.problem_type}')

    kwargs['pipeline_dtype'] = args.torch_dtype
    kwargs['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kwargs['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kwargs['fp8_param'] = args.fp8_param_gather
    swiglu = kwargs.get('swiglu', True)
    add_bias_linear = kwargs.get('add_bias_linear', False)
    num_moe_experts = kwargs.get('num_moe_experts', None)
    position_embedding_type = kwargs.get('position_embedding_type', 'rope')
    if position_embedding_type != 'rope':
        kwargs['apply_rope_fusion'] = False
    if not swiglu and not add_bias_linear:
        kwargs['bias_activation_fusion'] = False
    if add_bias_linear and num_moe_experts and args.moe_grouped_gemm:
        kwargs['bias_dropout_fusion'] = False
    if num_moe_experts is None:
        kwargs['expert_model_parallel_size'] = 1
        kwargs['expert_tensor_parallel_size'] = 1

    if args.router_replay_mode != 'disabled':
        kwargs['moe_enable_routing_replay'] = True
    config = ModelConfig(**kwargs)
    if is_torch_npu_available() and getattr(args, 'attention_backend', 'flash') != 'local':
        setattr(config, 'use_flash_attn', True)
    _check_attention_backend(args, config)
    _check_padding_free(args, config)
    return config


def get_mcore_model(args, hf_config):
    config = get_mcore_model_config(args, hf_config)
    models = _get_mcore_model(config)

    return models

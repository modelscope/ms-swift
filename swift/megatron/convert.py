# Copyright (c) ModelScope Contributors. All rights reserved.

import math
import os
import shutil
from dataclasses import fields

import torch
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint as mg_save_checkpoint
from megatron.training.initialize import initialize_megatron
from transformers.utils import strtobool

from swift.arguments import ExportArguments
from swift.pipelines import prepare_model_template
from swift.utils import get_logger, get_n_params_grads, is_master
from .arguments import MegatronArguments
from .model import get_megatron_model_meta
from .utils import convert_hf_config, patch_load_base_checkpoint, patch_torch_dist_shard, test_convert_precision

logger = get_logger()

convert_kwargs = {
    'use_cpu_initialization': True,
    'no_save_optim': True,
    'no_save_rng': True,
    'no_load_optim': True,
    'no_load_rng': True,
    'finetune': True,
    'attention_backend': 'unfused',
}


def _check_megatron_kwargs(kwargs):
    # Make sure that the keys in kwargs have default values of None in MegatronArguments.
    default_mapping = {field.name: field.default for field in fields(MegatronArguments)}
    for k in kwargs.keys():
        assert default_mapping[k] is None


def convert_hf2mcore(args: ExportArguments) -> None:
    hf_model, template = prepare_model_template(args, patch_offload=not args.test_convert_precision)
    processor = template.processor
    if args.thread_count is None:
        checkpoint_size = sum(get_n_params_grads(hf_model)[0]) * torch.finfo(args.torch_dtype).bits // 8e9
        args.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB
    patch_torch_dist_shard(args.thread_count)

    megatron_model_meta = get_megatron_model_meta(args.model_type)
    assert megatron_model_meta is not None, f'Model: {args.model} is not supported.'
    kwargs = convert_hf_config(processor.model_info.config)
    logger.info(f'megatron_config: {kwargs}')
    _check_megatron_kwargs(kwargs)
    current_convert_kwargs = convert_kwargs.copy()
    if args.model_info.is_moe_model:
        current_convert_kwargs['moe_grouped_gemm'] = True
    megatron_args = MegatronArguments(
        model=args.model,
        model_type=args.model_type,
        **kwargs,
        **current_convert_kwargs,
        save=args.output_dir,
        torch_dtype=args.torch_dtype)
    extra_args = megatron_args.parse_to_megatron()
    extra_args_provider = megatron_model_meta.extra_args_provider
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=extra_args)

    mg_model = megatron_model_meta.model_provider()
    logger.info('Megatron model created successfully.')
    bridge = megatron_model_meta.bridge_cls()
    bridge.load_weights(mg_model, args.model_info.model_dir)
    logger.info('Successfully transferred HF model weights to MG model.')
    _test_convert_precision = strtobool(os.getenv('SWIFT_TEST_CONVERT_PRECISION', '0'))
    if not _test_convert_precision:
        args.save_args()
        logger.info('Saving the model...')
        mg_save_checkpoint(1, [mg_model], None, None, 0)
        logger.info(f'Successfully saved Megatron model weights in `{args.output_dir}`.')
    # Place it at the end to avoid test_convert_precision affecting precision.
    if args.test_convert_precision:
        test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)


def convert_mcore2hf(args: ExportArguments) -> None:
    from swift.megatron import prepare_mcore_model, adapter_state_dict_context
    _, template = prepare_model_template(args, load_model=False)
    processor = template.processor

    megatron_model_meta = get_megatron_model_meta(args.model_type)
    assert megatron_model_meta is not None, f'Model: {args.model} is not supported.'
    hf_config = processor.model_info.config
    kwargs = convert_hf_config(hf_config)
    logger.info(f'megatron_config: {kwargs}')
    _check_megatron_kwargs(kwargs)
    current_convert_kwargs = convert_kwargs.copy()
    if args.model_info.is_moe_model:
        current_convert_kwargs['moe_grouped_gemm'] = True
    adapter_load = args.mcore_adapters[0] if args.mcore_adapters else None
    extra_config = MegatronArguments.load_args_config(adapter_load or args.mcore_model)
    extra_config['adapter_load'] = adapter_load
    if args.mcore_model is not None:
        extra_config['load'] = args.mcore_model
    kwargs.update(extra_config)
    megatron_args = MegatronArguments(
        model=args.model,
        model_type=args.model_type,
        **kwargs,
        **current_convert_kwargs,
        save=args.output_dir if args.to_mcore else None,
        torch_dtype=args.torch_dtype)
    extra_args = megatron_args.parse_to_megatron()
    extra_args_provider = megatron_model_meta.extra_args_provider
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=extra_args)

    mg_model = megatron_model_meta.model_provider()
    if megatron_args.load is None:
        raise ValueError('Please specify `--mcore_model`.')
    with patch_load_base_checkpoint():
        load_checkpoint([mg_model], None, None, strict=True)
    if megatron_args.adapter_load is not None:
        peft_model = prepare_mcore_model(mg_model)
        with adapter_state_dict_context():
            load_checkpoint([mg_model], None, None, load_arg='adapter_load', strict=False)
        logger.info('Merge LoRA...')
        mg_model = peft_model.merge_and_unload()
    logger.info('Megatron model created successfully.')
    if args.to_hf:
        bridge = megatron_model_meta.bridge_cls()
        logger.info('Converting weights and saving the model...')
        bridge.save_weights([mg_model], args.output_dir, processor=processor, config=hf_config)
        if is_master():
            args_path = os.path.join(megatron_args.adapter_load or megatron_args.load or args.model, 'args.json')
            if os.path.exists(args_path):
                shutil.copy(args_path, os.path.join(args.output_dir, 'args.json'))
            else:
                args.save_args(args.output_dir)
        logger.info(f'Successfully saved HF model weights in `{args.output_dir}`.')
        if args.test_convert_precision:
            hf_model, template = prepare_model_template(args, model=args.output_dir)
            test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
    elif args.to_mcore:
        if args.thread_count is None:
            checkpoint_size = sum(get_n_params_grads(mg_model)[0]) * torch.finfo(args.torch_dtype).bits // 8e9
            args.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB
        patch_torch_dist_shard(args.thread_count)

        args.save_args()
        logger.info('Saving the model...')
        mg_save_checkpoint(1, [mg_model], None, None, 0)
        logger.info(f'Successfully saved Megatron model weights in `{args.output_dir}`.')

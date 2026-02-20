# Copyright (c) ModelScope Contributors. All rights reserved.

import math
import os
import shutil

import torch
from transformers.utils import strtobool

from swift.arguments import ExportArguments
from swift.pipelines import prepare_model_template
from swift.utils import get_logger, get_n_params_grads, is_master
from .arguments import MegatronArguments
from .model import get_mcore_model
from .utils import (load_mcore_checkpoint, patch_torch_dist_shard, prepare_mcore_model, save_mcore_checkpoint,
                    test_convert_precision)

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


def convert_hf2mcore(args: ExportArguments) -> None:
    hf_model, template = prepare_model_template(args, patch_offload=not args.test_convert_precision)
    processor = template.processor
    if args.thread_count is None:
        checkpoint_size = sum(get_n_params_grads(hf_model)[0]) * torch.finfo(args.torch_dtype).bits // 8e9
        args.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB
    patch_torch_dist_shard(args.thread_count)

    hf_config = processor.model_info.config
    current_convert_kwargs = convert_kwargs.copy()
    if args.model_info.is_moe_model:
        current_convert_kwargs['moe_grouped_gemm'] = True
    megatron_args = MegatronArguments(
        model=args.model,
        model_type=args.model_type,
        **current_convert_kwargs,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype)

    mg_model = get_mcore_model(megatron_args, hf_config)[0]
    logger.info('Megatron model created successfully.')
    bridge = megatron_args.megatron_model_meta.bridge_cls(megatron_args)
    bridge.load_weights([mg_model], args.model_info.model_dir)
    logger.info('Successfully transferred HF model weights to MG model.')
    _test_convert_precision = strtobool(os.getenv('SWIFT_TEST_CONVERT_PRECISION', '0'))
    if not _test_convert_precision:
        args.save_args()
        logger.info('Saving the model...')
        save_mcore_checkpoint(megatron_args, [mg_model])
    # Place it at the end to avoid test_convert_precision affecting precision.
    if args.test_convert_precision:
        test_convert_precision(megatron_args, hf_model, mg_model, template, test_convert_dtype=args.test_convert_dtype)


def convert_mcore2hf(args: ExportArguments) -> None:
    _, template = prepare_model_template(args, load_model=False)
    processor = template.processor

    hf_config = processor.model_info.config
    current_convert_kwargs = convert_kwargs.copy()
    if args.model_info.is_moe_model:
        current_convert_kwargs['moe_grouped_gemm'] = True
    extra_config = MegatronArguments.load_args_config(args.mcore_adapter or args.mcore_model)
    extra_config['mcore_adapter'] = args.mcore_adapter
    if args.mcore_model is not None:
        extra_config['mcore_model'] = args.mcore_model
    current_convert_kwargs.update(extra_config)
    megatron_args = MegatronArguments(
        model=args.model,
        model_type=args.model_type,
        **current_convert_kwargs,
        output_dir=args.output_dir if args.to_mcore else None,
        torch_dtype=args.torch_dtype)

    mg_model = get_mcore_model(megatron_args, hf_config)[0]
    if megatron_args.mcore_model is None:
        raise ValueError('Please specify `--mcore_model`.')
    load_mcore_checkpoint(megatron_args, [mg_model], load_arg='mcore_model')
    if megatron_args.mcore_adapter is not None:
        peft_model = prepare_mcore_model(megatron_args, mg_model)
        load_mcore_checkpoint(megatron_args, [mg_model], load_arg='mcore_adapter')
        logger.info('Merge LoRA...')
        mg_model = peft_model.merge_and_unload()
    logger.info('Megatron model created successfully.')
    if args.to_hf:
        bridge = megatron_args.megatron_model_meta.bridge_cls(megatron_args)
        logger.info('Converting weights and saving the model...')
        bridge.save_weights([mg_model], args.output_dir, processor=processor, hf_config=hf_config)
        if is_master():
            args_path = os.path.join(megatron_args.mcore_adapter or megatron_args.mcore_model or args.model,
                                     'args.json')
            if os.path.exists(args_path):
                shutil.copy(args_path, os.path.join(args.output_dir, 'args.json'))
            else:
                args.save_args(args.output_dir)
        if args.test_convert_precision:
            hf_model, template = prepare_model_template(args, model=args.output_dir)
            test_convert_precision(
                megatron_args, hf_model, mg_model, template, test_convert_dtype=args.test_convert_dtype)
    elif args.to_mcore:
        if args.thread_count is None:
            checkpoint_size = sum(get_n_params_grads(mg_model)[0]) * torch.finfo(args.torch_dtype).bits // 8e9
            args.thread_count = max(math.ceil(checkpoint_size / 10), 2)  # 10GB
        patch_torch_dist_shard(args.thread_count)

        args.save_args()
        logger.info('Saving the model...')
        save_mcore_checkpoint(megatron_args, [mg_model])

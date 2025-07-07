# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.distributed as dist
from megatron.core import mpu
from megatron.training import get_args

from swift.utils import activate_parameters, freeze_parameters, get_logger, get_model_parameter_info

logger = get_logger()


def prepare_mcore_model(model) -> None:
    args = get_args()
    if args.train_type == 'full':
        freeze_parameters(model, args.freeze_parameters_ratio, args.freeze_parameters, args.freeze_parameters_regex)
        if args.trainable_parameters or args.trainable_parameters_regex:
            activate_parameters(model, args.trainable_parameters, args.trainable_parameters_regex)
    elif args.train_type == 'lora':
        from swift.tuners import LoraConfig, Swift
        target_modules = get_target_modules(args, model)
        modules_to_save = get_modules_to_save(args, model)
        lora_kwargs = {
            'r': args.lora_rank,
            'target_modules': target_modules,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'bias': args.lora_bias,
            'modules_to_save': modules_to_save,
            'use_rslora': args.use_rslora,
        }
        lora_config = LoraConfig(task_type='CAUSAL_LM', lora_dtype=args.lora_dtype, **lora_kwargs)
        model.prepare_inputs_for_generation = None  # fix error
        model = Swift.prepare_model(model, lora_config)
        logger.info(f'lora_config: {lora_config}')
    logger.info(f'model: {model}')
    logger.info_if(
        f'[rank{dist.get_rank()}] model_parameter_info: {get_model_parameter_info(model)}',
        cond=mpu.get_data_parallel_rank() == 0)

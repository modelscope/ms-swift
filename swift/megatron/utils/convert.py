# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
from megatron.training.initialize import initialize_megatron

from swift.llm import ExportArguments, get_model_tokenizer, save_checkpoint
from swift.utils import get_logger
from .argument import MegatronArguments
from .model import get_megatron_model_meta
from .utils import patch_megatron

logger = get_logger()


def convert_hf2megatron(args: ExportArguments) -> None:
    kwargs = args.get_model_kwargs()
    kwargs['torch_dtype'] = torch.float32
    hf_model, processor = get_model_tokenizer(**kwargs)
    megatron_model_meta = get_megatron_model_meta(args.model)
    kwargs = megatron_model_meta.load_config(processor.model_info.config)
    kwargs.update({
        'use_cpu_initialization': True,
        'torch_dtype': args.torch_dtype,
        'load': args.model_dir,
        'save': args.output_dir
    })
    megatron_args = MegatronArguments(**kwargs)
    patch_megatron(processor)
    extra_args = megatron_args.parse_to_megatron()
    initialize_megatron(args_defaults=extra_args)

    mg_model = megatron_model_meta.get_model_provider()()
    megatron_model_meta.convert_hf2megatron(hf_model, mg_model)
    args.save_args()
    logger.info(f'Successfully converted HF format to Megatron format and saved in `{args.output_dir}`.')


def convert_megatron2hf(args: ExportArguments) -> None:
    kwargs = args.get_model_kwargs()
    kwargs['torch_dtype'] = torch.float32
    hf_model, processor = get_model_tokenizer(**kwargs)
    megatron_model_meta = get_megatron_model_meta(args.model)
    kwargs = megatron_model_meta.load_config(processor.model_info.config)
    kwargs.update({
        'seq_length': 1,
        'use_cpu_initialization': True,
        'load': args.megatron_model,
        'save': args.output_dir,
        'hf_ckpt_path': args.model_dir
    })
    megatron_args = MegatronArguments(**kwargs, **MegatronArguments.get_matched_kwargs(args))
    patch_megatron(processor)
    extra_args = megatron_args.parse_to_megatron()
    initialize_megatron(args_defaults=extra_args)

    megatron_model_meta.convert_megatron2hf(hf_model, megatron_model_meta.get_model_provider())
    if args.torch_dtype is not None:
        hf_model.to(args.torch_dtype)
    save_checkpoint(
        hf_model,
        processor,
        args.output_dir,
        safe_serialization=args.safe_serialization,
        model_dirs=[args.megatron_model, args.model_dir],
        max_shard_size=args.max_shard_size,
        additional_saved_files=hf_model.model_meta.additional_saved_files)
    args.save_args()
    logger.info(f'Successfully converted Megatron format to HF format and saved in `{args.output_dir}`.')

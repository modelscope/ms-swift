# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import torch
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

from swift.llm import ExportArguments, get_model_tokenizer
from .argument import MegatronArguments
from .model import get_megatron_model_meta
from .utils import patch_megatron


def convert_hf2megatron(args: ExportArguments) -> None:
    args.save_args()
    hf_model, processor = get_model_tokenizer(**args.get_model_kwargs())
    megatron_model_meta = get_megatron_model_meta(args.model)
    kwargs = megatron_model_meta.load_config(hf_model.model_info)
    kwargs.update({'seq_length': 1, 'use_cpu_initialization': True, 'load': args.model_dir, 'save': args.output_dir})
    megatron_args = MegatronArguments(**kwargs, **MegatronArguments.get_matched_kwargs(args))
    patch_megatron(processor)
    extra_args = megatron_args.parse_to_megatron()
    initialize_megatron(args_defaults=extra_args)

    mg_model = megatron_model_meta.get_model_provider()()
    megatron_model_meta.convert_hf2megatron(hf_model, mg_model)


def convert_megatron2hf(args: ExportArguments) -> None:
    from swift.llm import save_checkpoint
    args.save_args()
    hf_model, processor = get_model_tokenizer(**args.get_model_kwargs())
    megatron_model_meta = get_megatron_model_meta(args.model)
    kwargs = megatron_model_meta.load_config(hf_model.model_info)
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

    mg_model = megatron_model_meta.get_model_provider()()
    megatron_model_meta.convert_megatron2hf(hf_model, mg_model)
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

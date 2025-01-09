# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import torch

from swift.llm import ExportArguments, get_model_tokenizer
from .model import get_megatron_model_meta


def convert_hf2megatron(args: ExportArguments) -> None:

    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    kwargs = args.get_model_kwargs()
    kwargs['torch_dtype'] = torch.float32
    hf_model, processor = get_model_tokenizer(**kwargs)
    megatron_model_meta = get_megatron_model_meta(args.model)
    model_provider = megatron_model_meta.get_model_provider()
    megatron_model_meta.load_config(hf_model.model_info)

    initialize_megatron(args_defaults=extra_args)
    args = get_args()
    model_provider, convert_module = get_megatron_model_convert(args.model_type)
    mg_model = model_provider()
    convert_module.convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    if save_torch_dtype is not None:
        mg_model.to(save_torch_dtype)
    convert_module.save_mgmodel(mg_model, args)


def convert_megatron2hf(
    hf_model,
    extra_args: Dict[str, Any],
) -> None:
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    initialize_megatron(args_defaults=extra_args)
    args = get_args()

    model_provider, convert_module = get_megatron_model_convert(args.model_type)
    convert_module.model_provider = model_provider
    mg_model = convert_module.load_megatron_model(args)  # no copy
    convert_module.convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)

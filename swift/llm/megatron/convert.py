# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict, Optional

import torch

from .model import get_megatron_model_convert


def convert_hf_to_megatron(
    hf_model,
    extra_args: Dict[str, Any],
    save_torch_dtype: Optional[torch.dtype] = None,
) -> None:
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    initialize_megatron(args_defaults=extra_args)
    args = get_args()

    model_provider, convert_module = get_megatron_model_convert(args.model_type)
    mg_model = model_provider()
    convert_module.convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    if save_torch_dtype is not None:
        mg_model.to(save_torch_dtype)
    convert_module.save_mgmodel(mg_model, args)


def convert_megatron_to_hf(
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

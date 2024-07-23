# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict, Optional

import torch

from .model import model_provider

_convert_mapping = {'qwen2': 'qwen'}


def convert_hf_to_megatron(
    hf_model,
    extra_args: Dict[str, Any],
    check_model_forward: bool = False,
    save_torch_dtype: Optional[torch.dtype] = None,
) -> None:
    megatron_patch_path = os.environ['PAI_MEGATRON_PATCH_PATH']
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    initialize_megatron(args_defaults=extra_args)
    args = get_args()

    sys.path.append(
        os.path.join(megatron_patch_path, 'toolkits/model_checkpoints_convertor', _convert_mapping[args.model_series]))
    from hf2mcore_qwen2_dense_and_moe_gqa import (convert_checkpoint_from_transformers_to_megatron, check_hf_mg_forward,
                                                  save_mgmodel)
    mg_model = model_provider()
    convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    if save_torch_dtype is not None:
        mg_model.to(save_torch_dtype)
    if check_model_forward:
        if save_torch_dtype is not None:
            hf_model.to(save_torch_dtype)
        check_hf_mg_forward(hf_model, mg_model, args)
    if args.load is None:
        args.load = hf_model.model_dir
    save_mgmodel(mg_model, args)


def convert_megatron_to_hf(
    hf_model,
    extra_args: Dict[str, Any],
) -> None:
    megatron_patch_path = os.environ['PAI_MEGATRON_PATCH_PATH']
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    initialize_megatron(args_defaults=extra_args)
    args = get_args()
    sys.path.append(
        os.path.join(megatron_patch_path, 'toolkits/model_checkpoints_convertor', _convert_mapping[args.model_series]))

    from hf2mcore_qwen2_dense_and_moe_gqa import convert_checkpoint_from_megatron_to_transformers, load_megatron_model
    import hf2mcore_qwen2_dense_and_moe_gqa
    hf2mcore_qwen2_dense_and_moe_gqa.model_provider = model_provider
    mg_model = load_megatron_model(args)
    convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)

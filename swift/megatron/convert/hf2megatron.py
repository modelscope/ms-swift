# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
from swift.llm import get_model_tokenizer, ExportArguments

from ..model import get_megatron_model_meta


def convert_hf2megatron(
    args: ExportArguments
) -> None:

    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    kwargs = args.get_model_kwargs()
    kwargs['torch_dtype'] = torch.float32
    hf_model, processor = get_model_tokenizer(**kwargs)
    megatron_model_meta = get_megatron_model_meta(args.model)
    megatron_model_meta.get_model_provider()
    megatron_model_meta.load_config(hf_model.model_info)


    initialize_megatron(args_defaults=extra_args)
    args = get_args()
    model_provider, convert_module = get_megatron_model_convert(args.model_type)
    mg_model = model_provider()
    convert_module.convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    if save_torch_dtype is not None:
        mg_model.to(save_torch_dtype)
    convert_module.save_mgmodel(mg_model, args)



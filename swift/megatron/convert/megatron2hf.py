# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict


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

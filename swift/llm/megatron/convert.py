# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict, Optional

import torch

_convert_mapping = {'qwen2': 'qwen'}


def model_provider(pre_process=True, post_process=True):
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron_patch.model.qwen2.transformer_config import Qwen2TransformerConfig
    from megatron_patch.model.qwen2.layer_specs import (get_gpt_layer_local_spec,
                                                        get_gpt_layer_with_transformer_engine_spec)
    from megatron_patch.model.qwen2.model import GPTModel

    args = get_args()
    config = core_transformer_config_from_args(args, Qwen2TransformerConfig)
    use_te = args.transformer_impl == 'transformer_engine'

    if use_te:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm,
                                                                            args.qk_layernorm)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor)
    return model


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

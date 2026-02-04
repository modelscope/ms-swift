# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from typing import TYPE_CHECKING, Optional, Union

import megatron.core
import torch
from megatron.core.models.gpt.gpt_layer_specs import (get_gpt_decoder_block_spec, get_gpt_layer_local_spec,
                                                      get_gpt_layer_with_transformer_engine_spec,
                                                      get_gpt_mtp_block_spec)
from packaging import version

from swift.megatron.utils import core_transformer_config_from_args, convert_hf_config
from swift.utils import get_logger

logger = get_logger()

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')

if TYPE_CHECKING:
    from .gpt_model import GPTModel


def _get_transformer_layer_spec(args):
    kwargs = {'qk_l2_norm': args.qk_l2_norm} if mcore_013 else {}
    return get_gpt_layer_with_transformer_engine_spec(
        args.num_experts,
        args.moe_grouped_gemm,
        args.qk_layernorm,
        args.multi_latent_attention,
        **kwargs,
    )


def get_mcore_model(args, model_args, pre_process=True, post_process=True, vp_stage: Optional[int] = None) -> 'GPTModel':
    from .register import get_megatron_model_meta
    megatron_model_meta = get_megatron_model_meta(args.model_type)

    logger.info('building GPT model ...')
    config = core_transformer_config_from_args(args, model_args)
    config.variable_seq_lengths = True
    if megatron_model_meta.get_transformer_layer_spec is not None:
        transformer_layer_spec = megatron_model_meta.get_transformer_layer_spec(config, vp_stage=vp_stage)
    else:
        if args.num_experts:
            kwargs = {'qk_l2_norm': args.qk_l2_norm, 'vp_stage': vp_stage} if mcore_013 else {}
            # Define the decoder block spec
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config, use_transformer_engine=True, normalization=args.normalization, **kwargs)
        else:
            # Define the decoder layer spec
            transformer_layer_spec = _get_transformer_layer_spec(args)
    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            transformer_layer_spec_for_mtp = _get_transformer_layer_spec(args)
        else:
            transformer_layer_spec_for_mtp = transformer_layer_spec
        kwargs = {'vp_stage': vp_stage} if mcore_013 else {}
        if megatron_model_meta.get_mtp_block_spec is not None:
            get_mtp_block_spec = megatron_model_meta.get_mtp_block_spec
        else:
            get_mtp_block_spec = get_gpt_mtp_block_spec
        mtp_block_spec = get_mtp_block_spec(
            config, transformer_layer_spec_for_mtp, use_transformer_engine=True, **kwargs)

    if args.use_shared_expert_gate and args.num_experts and args.moe_shared_expert_intermediate_size:
        for layer_spec in transformer_layer_spec.layer_specs:
            if hasattr(layer_spec.submodules.mlp.submodules, 'shared_experts'):
                layer_spec.submodules.mlp.submodules.shared_experts.params = {'gate': True}
    model = megatron_model_meta.model_cls(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=math.ceil(args.padded_vocab_size / args.tensor_model_parallel_size)
        * args.tensor_model_parallel_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_base=args.rotary_base,
        hf_rope_scaling=args.rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )

    return model

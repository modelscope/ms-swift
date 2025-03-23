# Copyright (c) Alibaba, Inc. and its affiliates.
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from ..rope import update_rope_inv_freq


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    config.variable_seq_lengths = True
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm,
                                                                        args.qk_layernorm, args.multi_latent_attention)
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
        rope_scaling=args.use_rope_scaling,
        rope_scaling_factor=args.rope_scaling_factor,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor)
    if args.rope_scaling:
        update_rope_inv_freq(model.rotary_pos_emb.inv_freq, args.rope_scaling)
    return model

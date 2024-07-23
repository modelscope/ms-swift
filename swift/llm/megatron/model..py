# Copyright (c) Alibaba, Inc. and its affiliates.
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

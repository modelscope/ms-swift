# Copyright (c) Alibaba, Inc. and its affiliates.

def get_model_provider(gpt_model_cls, transformer_config_cls, layer_spec_module):
    def model_provider(pre_process=True, post_process=True):
        from megatron.training import get_args
        from megatron.training.arguments import core_transformer_config_from_args
        args = get_args()
        config = core_transformer_config_from_args(args, transformer_config_cls)
        transformer_layer_spec = layer_spec_module.get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        model = gpt_model_cls(
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
    return model_provider


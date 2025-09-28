# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING, Optional, Union

import megatron.legacy
import torch
from megatron.core.models.gpt.gpt_layer_specs import (get_gpt_decoder_block_spec, get_gpt_layer_local_spec,
                                                      get_gpt_layer_with_transformer_engine_spec,
                                                      get_gpt_mtp_block_spec)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import get_gpt_heterogeneous_layer_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

if TYPE_CHECKING:
    from .gpt_model import GPTModel
    from .mm_gpt_model import MultimodalGPTModel


# Code borrowed from NVIDIA/Megatron-LM
def model_provider(
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None) -> Union['GPTModel', 'MultimodalGPTModel', megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == 'transformer_engine'

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f'oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}', 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, 'language_model')
    else:
        config = core_transformer_config_from_args(args)
    config.variable_seq_lengths = True
    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        elif args.megatron_model_meta.get_transformer_layer_spec is not None:
            transformer_layer_spec = args.megatron_model_meta.get_transformer_layer_spec(config, vp_stage=vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config,
                    use_transformer_engine=use_te,
                    normalization=args.normalization,
                    qk_l2_norm=args.qk_l2_norm,
                    vp_stage=vp_stage)
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=use_te, vp_stage=vp_stage)

        if args.use_shared_expert_gate and args.num_experts and args.moe_shared_expert_intermediate_size:
            # qwen2_moe
            for layer_spec in transformer_layer_spec.layer_specs:
                layer_spec.submodules.mlp.submodules.shared_experts.params = {'gate': True}
        model = args.megatron_model_meta.model_cls(
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
            hf_rope_scaling=args.rope_scaling,
            rope_scaling=args.use_rope_scaling,
            rope_scaling_factor=args.rope_scaling_factor,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    return model

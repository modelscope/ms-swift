# Copyright (c) ModelScope Contributors. All rights reserved.
# Parts of the functions in this file are code borrowed from NVIDIA/Megatron-LM
import dataclasses
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer import MLATransformerConfig, TransformerConfig

from swift.utils import get_logger, is_master, seed_everything

logger = get_logger()


@contextmanager
def _patch_megatron_timeout(distributed_timeout_minutes):
    from megatron.core import parallel_state

    origin_create_group = parallel_state.create_group

    def create_group(ranks=None, timeout=None, *_args, **kwargs):
        if timeout is None:
            timeout = timedelta(minutes=distributed_timeout_minutes)
        return origin_create_group(ranks, timeout, *_args, **kwargs)

    parallel_state.create_group = create_group
    try:
        yield
    finally:
        parallel_state.create_group = origin_create_group


def _initialize_mpu(args):
    """Initialize torch.distributed and core model parallel."""
    if not torch.distributed.is_initialized():
        raise ValueError('torch.distributed is not initialized')
    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()

    if mpu.model_parallel_is_initialized():
        logger.info('model parallel is already initialized')
    else:
        with _patch_megatron_timeout(args.distributed_timeout_minutes):
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                context_parallel_size=args.context_parallel_size,
                expert_model_parallel_size=args.expert_model_parallel_size,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
            )
        if is_master():
            logger.info(f'tp: {args.tensor_model_parallel_size}, pp: {args.pipeline_model_parallel_size}, '
                        f'vpp: {args.virtual_pipeline_model_parallel_size}, cp: {args.context_parallel_size}, '
                        f'ep: {args.expert_model_parallel_size}, etp: {args.expert_tensor_parallel_size}')


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = True,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (1009 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (11 * mpu.get_data_parallel_rank())
        seed_everything(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker,
                                                            use_cudagraphable_rng)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed_))


def initialize_megatron(args):
    # Pytorch distributed.
    _initialize_mpu(args)

    # Random seeds for reproducibility.
    logger.info(f'Setting random seeds to {args.seed}.')
    _set_random_seed(args.seed)

    # Setup MoE aux loss scale value.
    if args.num_experts is not None:
        from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

    # TODO: tp_comm_overlap, _compile_dependencies


def core_transformer_config_from_args(args, config_class=None):
    # Config class.
    config_class = config_class or TransformerConfig

    if args.multi_latent_attention:
        config_class = MLATransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = True
    # TODO: apply_layernorm_1p
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.torch_dtype
    kw_args['batch_p2p_comm'] = True
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kw_args['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kw_args['fp8_param'] = args.fp8_param_gather
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.quick_geglu:
        assert not args.swiglu
        kw_args['gated_linear_unit'] = True
        kw_args['activation_func'] = quick_gelu
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None

    kw_args['cp_comm_type'] = 'p2p'
    kw_args['inference_sampling_seed'] = args.seed

    config = config_class(**kw_args)
    config.args = args

    return config

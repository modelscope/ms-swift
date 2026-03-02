# Copyright (c) ModelScope Contributors. All rights reserved.
# Parts of the functions in this file are code borrowed from NVIDIA/Megatron-LM
import copy
import dataclasses
import megatron.core
import numpy as np
import os
import random
import torch
from argparse import Namespace
from contextlib import contextmanager
from datetime import timedelta
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import (get_default_load_sharded_strategy,
                                                            get_default_save_sharded_strategy)
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest
from megatron.core.dist_checkpointing.strategies.fully_parallel import (FullyParallelLoadStrategyWrapper,
                                                                        FullyParallelSaveStrategyWrapper)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train
from megatron.core.fusions.fused_bias_gelu import bias_gelu
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_torch_version, is_te_min_version, is_torch_min_version
from packaging import version
from typing import Optional

from swift.utils import check_json_format, get_logger, init_process_group, is_master, seed_everything, set_device
from .patcher import patch_merge_fn

logger = get_logger()

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


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
        set_device()
        init_process_group(args.ddp_backend, args.ddp_timeout)
    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()

    if mpu.model_parallel_is_initialized():
        logger.info('model parallel is already initialized')
    else:
        distributed_timeout_minutes = args.ddp_timeout // 60
        with _patch_megatron_timeout(distributed_timeout_minutes):
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                context_parallel_size=args.context_parallel_size,
                expert_model_parallel_size=args.expert_model_parallel_size,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=distributed_timeout_minutes,
            )
        if is_master():
            logger.info(f'TP: {args.tensor_model_parallel_size}, PP: {args.pipeline_model_parallel_size}, '
                        f'VPP: {args.virtual_pipeline_model_parallel_size}, CP: {args.context_parallel_size}, '
                        f'EP: {args.expert_model_parallel_size}, ETP: {args.expert_tensor_parallel_size}')


def set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
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
    set_random_seed(args.seed, args.data_parallel_random_init, args.te_rng_tracker)

    # Setup MoE aux loss scale value.
    if args.model_info.is_moe_model:
        from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))


def _get_rng_state():
    """Collect rng state across data parallel ranks."""
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()
    }

    # data_parallel_random_init False
    rng_state_list = [rng_state]

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    rng_state_list = ShardedObject(
        'rng_state',
        rng_state_list, (pp_size, tp_size), (pp_rank, tp_rank),
        replica_id=mpu.get_data_parallel_rank(with_context_parallel=True))
    return rng_state_list


def _generate_state_dict(args,
                         models,
                         optimizer=None,
                         opt_param_scheduler=None,
                         rng_state=None,
                         iteration=None,
                         model_sd_kwargs=None,
                         optim_sd_kwargs=None):
    model_sd_kwargs = model_sd_kwargs or {}
    state_dict = {
        'args': Namespace(**check_json_format(vars(args))),
        'checkpoint_version': 3.0,
    }
    if iteration is not None:
        state_dict['iteration'] = iteration
    for i, m in enumerate(models):
        key = 'model'
        if len(models) > 1:
            key = f'model{i}'
        model_sd = models[i].sharded_state_dict(**model_sd_kwargs)
        state_dict[key] = model_sd

    if not args.no_save_optim:
        if not mcore_013:
            optim_sd_kwargs = None
        if optimizer is not None:
            state_dict['optimizer'] = optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
        if opt_param_scheduler is not None:
            state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()

    if not args.no_save_rng and rng_state is not None:
        state_dict['rng_state'] = rng_state
    return state_dict


def _filter_adapter_state_dict(state_dict, is_peft_format: bool, adapter_name: str = 'default'):
    """
    When is_peft_format is True, keep only the PEFT format state_dict;
    when False, remove the PEFT format state_dict.

    This function ensures it is called when tuner_type != 'full'.
    """
    if 'model' in state_dict:
        n_models = 1
    else:
        n_models = 0
        while f'model{n_models}' in state_dict:
            n_models += 1

    for i in range(n_models):
        if i == 0 and n_models == 1:
            model_key = 'model'
        else:
            model_key = f'model{i}'
        new_state_dict = {}
        state_dict_model = state_dict[model_key]
        for k, v in state_dict_model.items():
            if is_peft_format:
                if '.lora_A.' in k or '.lora_B.' in k or '.modules_to_save.' in k:
                    new_state_dict[k] = v
            else:
                if '.lora_A.' in k or '.lora_B.' in k or 'original_module.' in k:
                    continue
                k = k.replace('base_layer.', '')
                k = k.replace(f'modules_to_save.{adapter_name}.', '')
                v.key = v.key.replace('base_layer.', '')
                v.key = v.key.replace(f'modules_to_save.{adapter_name}.', '')
                new_state_dict[k] = v
        state_dict[model_key] = new_state_dict


def _preprocess_common_before_consistancy_check(common_state_dict):
    # Convert args key of type namespace to dictionary
    preprocessed_common_state_dict = copy.deepcopy(common_state_dict)
    preprocessed_common_state_dict['args'] = vars(preprocessed_common_state_dict['args'])
    # Remove rank and local rank from state dict if it exists, since they are expected to be different
    preprocessed_common_state_dict['args'].pop('local_rank', None)
    preprocessed_common_state_dict['args'].pop('rank', None)
    return preprocessed_common_state_dict


def get_sharded_sd_metadata(args):
    sharded_sd_metadata = {'singleton_local_shards': False, 'chained_optim_avoid_prefix': True}
    force_pre_mcore_014 = not is_torch_min_version('2.6a0')
    if force_pre_mcore_014 and not args.dist_ckpt_save_pre_mcore_014:
        args.dist_ckpt_save_pre_mcore_014 = True
        logger.warning(f'PyTorch version {get_torch_version()} below 2.6 detected.'
                       f' Forcing dist_ckpt_save_pre_mcore_014 behavior.')

    if args.dist_ckpt_save_pre_mcore_014:
        sharded_sd_metadata['distrib_optim_sharding_type'] = 'fully_sharded_model_space'
    else:
        if args.dist_ckpt_optim_fully_reshardable:
            sharded_sd_metadata['distrib_optim_sharding_type'] = 'fully_reshardable'
            sharded_sd_metadata[
                'distrib_optim_fully_reshardable_mem_efficient'] = args.distrib_optim_fully_reshardable_mem_efficient
        else:
            sharded_sd_metadata['distrib_optim_sharding_type'] = 'dp_reshardable'
    return sharded_sd_metadata


def save_mcore_checkpoint(
    args,
    models,
    optimizer=None,
    opt_param_scheduler=None,
    iteration=1,
    output_dir: Optional[str] = None,
    is_peft_format: bool = False,
):
    if output_dir is None:
        output_dir = args.output_dir
    models = unwrap_model(models)
    rng_state = _get_rng_state() if models else None
    checkpoint_dir = os.path.join(output_dir, f'iter_{iteration:07d}')
    sharded_sd_metadata = get_sharded_sd_metadata(args)
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_dict = _generate_state_dict(
        args,
        models,
        optimizer,
        opt_param_scheduler,
        rng_state,
        iteration=iteration,
        model_sd_kwargs={'metadata': sharded_sd_metadata},
        optim_sd_kwargs={'metadata': sharded_sd_metadata},
    )
    _filter_adapter_state_dict(state_dict, is_peft_format)

    save_strategy = get_default_save_sharded_strategy()
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy,
        mpu.get_data_parallel_group(with_context_parallel=True),
    )
    kwargs = {'content_metadata': sharded_sd_metadata} if mcore_013 else {}
    async_save_request = dist_checkpointing.save(
        state_dict,
        checkpoint_dir,
        save_strategy,
        async_sharded_save=args.async_save,
        validate_access_integrity=True,
        preprocess_common_before_consistancy_check=_preprocess_common_before_consistancy_check,
        **kwargs)
    tracker_path = os.path.join(output_dir, 'latest_checkpointed_iteration.txt')
    try:
        from megatron.core.msc_utils import open_file
    except ImportError:
        open_file = open
    with open_file(tracker_path, 'w') as f:
        f.write(str(iteration))

    if not args.async_save:
        assert async_save_request is None
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if is_master():

        def iter_finalize_fn():
            if models:
                logger.info(f'Successfully saved Megatron model weights in `{output_dir}`.')

        if args.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(iter_finalize_fn)
        else:
            iter_finalize_fn()

    if args.async_save:
        schedule_async_save(async_save_request)


# Singleton manager of async calls
# The default is `TemporalAsyncCaller`
_async_calls_queue = AsyncCallsQueue()


def init_persistent_async_worker():
    global _async_calls_queue
    # Recreate the async_calls_queue for persistent worker
    # This duplicate step is for backward compatiblity
    _async_calls_queue = AsyncCallsQueue(persistent=True)


def schedule_async_save(async_request: AsyncRequest):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    _async_calls_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(args, blocking: bool = False, terminate=False):
    """Finalizes active async save calls.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    if not args.async_save:
        return

    _async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=False)

    if terminate:
        _async_calls_queue.close()


def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks."""
    return _async_calls_queue.get_num_unfinalized_calls() == 0


def _load_iteration(tracker_path: str):
    if not os.path.exists(tracker_path):
        return 0
    with open(tracker_path, 'r') as f:
        iteration = int(f.read())
    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.tensor([iteration], dtype=torch.long, device='cuda')
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        iteration = iters_cuda[0].item()
    return iteration


def load_mcore_checkpoint(args,
                          ddp_models: list,
                          optimizer=None,
                          opt_param_scheduler=None,
                          load_arg: str = 'mcore_model',
                          adapter_name: str = 'default'):
    if load_arg in {'mcore_adapter', 'mcore_ref_adapter'}:
        is_peft_format = True
    else:
        # 'mcore_model', 'mcore_ref_model'
        is_peft_format = False
    load_dir = getattr(args, load_arg)

    no_load_optim = args.no_load_optim
    no_load_rng = args.no_load_rng
    finetune = args.finetune
    if not is_peft_format and args.tuner_type != 'full':
        no_load_optim = True
        no_load_rng = True
        finetune = False
    models = unwrap_model(ddp_models)
    tracker_path = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    iteration = _load_iteration(tracker_path)
    checkpoint_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
    state_dict = dist_checkpointing.load_common_state_dict(checkpoint_dir)

    ckpt_tp_pp = (
        state_dict['args'].tensor_model_parallel_size,
        state_dict['args'].pipeline_model_parallel_size,
    )
    run_tp_pp = (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
    )
    mismatch_msg = f'(TP, PP) mismatch after resume ({run_tp_pp} vs {ckpt_tp_pp} from checkpoint)'
    # Determine if RNG state will be loaded
    if (ckpt_tp_pp == run_tp_pp and not finetune and not no_load_rng
            and not getattr(state_dict['args'], 'no_save_rng', False)):
        gen_sd_rng_state = _get_rng_state()  # we can load the rng state
    else:
        gen_sd_rng_state = None
        if ckpt_tp_pp != run_tp_pp:
            logger.info(f'{mismatch_msg}: RNG state will be ignored')
    sharded_sd_metadata = state_dict.get('content_metadata')
    if (not finetune and not no_load_optim and not getattr(state_dict['args'], 'no_save_optim', False)):
        gen_sd_optim = optimizer
        gen_sd_opt_param_scheduler = opt_param_scheduler

        if (args.use_distributed_optimizer and ckpt_tp_pp != run_tp_pp
                and (sharded_sd_metadata or {}).get('distrib_optim_sharding_type') not in {
                    'fully_reshardable',
                    'fully_sharded_model_space',
                    'fsdp_dtensor',
                }):
            raise RuntimeError(f'{mismatch_msg}: not supported for DistributedOptimizer')
    else:
        gen_sd_optim, gen_sd_opt_param_scheduler = None, None
    optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
    model_sd_kwargs = dict(metadata=sharded_sd_metadata)
    # TODO: check no_save_optim
    sharded_state_dict = _generate_state_dict(
        args,
        models,
        gen_sd_optim,
        gen_sd_opt_param_scheduler,
        gen_sd_rng_state,
        iteration=iteration,
        model_sd_kwargs=model_sd_kwargs,
        optim_sd_kwargs=optim_sd_kwargs)
    _filter_adapter_state_dict(sharded_state_dict, is_peft_format, adapter_name=adapter_name)
    model_keys = [k for k in sharded_state_dict.keys() if k.startswith('model')]  # compat vpp
    for k in model_keys:
        patch_merge_fn(sharded_state_dict[k])
    load_strategy = get_default_load_sharded_strategy(checkpoint_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(load_strategy,
                                                     mpu.get_data_parallel_group(with_context_parallel=True))
    state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_dir, load_strategy)

    if finetune:
        iteration = 0
    if 'args' in state_dict and not finetune:
        args.consumed_train_samples = getattr(state_dict['args'], 'consumed_train_samples', 0)

    if len(ddp_models) == 1:
        ddp_models[0].load_state_dict(state_dict['model'], strict=False)
    else:
        for i, m in enumerate(ddp_models):
            if f'model{i}' not in state_dict:
                continue
            m.load_state_dict(state_dict[f'model{i}'])

    if not finetune and not no_load_optim:
        if optimizer is not None:
            optimizer.load_state_dict(state_dict['optimizer'])
        if opt_param_scheduler is not None:
            opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
    elif (args.fp16 or args.bf16) and optimizer is not None:
        optimizer.reload_model_params()

    if not finetune and not no_load_rng:
        if 'rng_state' in state_dict:
            rng_state = state_dict['rng_state']
            if args.data_parallel_random_init:
                rng_state = rng_state[mpu.get_data_parallel_rank()]
            else:
                rng_state = rng_state[0]
            random.setstate(rng_state['random_rng_state'])
            np.random.set_state(rng_state['np_rng_state'])
            torch.set_rng_state(rng_state['torch_rng_state'])
            torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
            tensor_parallel.get_cuda_rng_tracker().set_states(rng_state['rng_tracker_states'])
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f'Successfully loaded Megatron model weights from: {load_dir}')
    return iteration


def wrap_model(args, models, wrap_with_ddp: bool = True):
    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for m in models:
        for param in m.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        if not args.use_cpu_initialization:
            m.cuda(torch.cuda.current_device())
    # Fp16
    config = models[0].config
    if args.fp16 or args.bf16:
        models = [Float16Module(config, model_module) for model_module in models]

    # DDP
    if not wrap_with_ddp:
        return
    kwargs = {}
    for f in dataclasses.fields(DistributedDataParallelConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    kwargs['check_for_nan_in_grad'] = True
    ddp_config = DistributedDataParallelConfig(**kwargs)

    # In the Megatron FSDP and DDP use path, we need to initialize the bucket size.
    # If bucket_size is not provided as an input, use sane default.
    # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
    # ring-reduce implementations are large enough to remain bandwidth-bound rather than
    # latency-bound.
    if ddp_config.bucket_size is None:
        ddp_config.bucket_size = max(40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True))
    # Set bucket_size to infinity if overlap_grad_reduce is False.
    if not ddp_config.overlap_grad_reduce:
        ddp_config.bucket_size = None

    with torch.cuda.stream(torch.cuda.Stream()):
        models = [
            DDP(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step,
            ) for (model_chunk_idx, model_chunk) in enumerate(models)
        ]

    # Broadcast params from data parallel src rank to other data parallel ranks.
    if args.data_parallel_random_init:
        for m in models:
            m.broadcast_params()

    return models


def get_optimizer_param_scheduler(args, optimizer):
    # Iteration-based training.
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def unwrap_model(models, module_instances=None):
    """Unwrap_model to return the final model instance"""
    try:
        from megatron.core.utils import unwrap_model
        return unwrap_model(models, module_instances)
    except ImportError:
        pass
    if module_instances is None:
        from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP
        module_instances = (DDP, torch_FSDP, Float16Module)

    return_list = True
    if not isinstance(models, list):
        models = [models]
        return_list = False
    unwrapped_model = []
    for model in models:
        while isinstance(model, module_instances):
            model = model.module
        unwrapped_model.append(model)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def should_disable_forward_pre_hook(args):
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def enable_forward_pre_hook(model_chunks):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks, param_sync=True):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def initialize_tp_communicators(args, config):
    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""
    from transformer_engine.pytorch import module as te_module
    input_shape = [
        (args.seq_length * args.micro_batch_size) // args.context_parallel_size,
        config.hidden_size,
    ]

    if is_te_min_version('2.7.0'):
        UserBufferQuantizationMode = te_module.base.UserBufferQuantizationMode
        quantization_modes = [UserBufferQuantizationMode.FP8 if args.fp8 else UserBufferQuantizationMode.NONE]
        if args.fp8 is not None and args.first_last_layers_bf16 and (args.num_layers_at_start_in_bf16 > 0
                                                                     or args.num_layers_at_end_in_bf16 > 0):
            quantization_modes.append(UserBufferQuantizationMode.NONE)
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            quantization_modes=quantization_modes,
        )
    elif is_te_min_version('1.9.0'):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            use_fp8=(args.fp8 is not None),
        )


def warmup_jit_function(config, args):
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    bias = torch.rand(config.ffn_hidden_size // config.tensor_model_parallel_size, dtype=dtype, device='cuda')
    input_tensor = torch.rand(
        (
            args.seq_length // config.context_parallel_size,
            args.micro_batch_size,
            config.ffn_hidden_size // config.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device='cuda',
    )
    # Warmup JIT fusions with the input_tensor grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input_tensor.requires_grad = bias_grad, input_grad
        for _ in range(5):
            if config.swiglu:
                output = bias_swiglu(input_tensor, bias)
            else:
                output = bias_gelu(bias, input_tensor)
    del bias, input_tensor, output

    # Warmup fused bias+dropout+add
    if config.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input_tensor = torch.rand(
        (seq_length // config.context_parallel_size, args.micro_batch_size, config.hidden_size),
        dtype=dtype,
        device='cuda',
    )
    residual = torch.rand(
        (seq_length // config.context_parallel_size, args.micro_batch_size, config.hidden_size),
        dtype=dtype,
        device='cuda',
    )
    bias = torch.rand((config.hidden_size), dtype=dtype, device='cuda').expand_as(residual)
    dropout_rate = 0.1
    # Warmup JIT fusions with the input_tensor grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        input_tensor.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train([input_tensor, bias], residual, dropout_rate)
    del bias, input_tensor, residual, output
    torch.cuda.empty_cache()

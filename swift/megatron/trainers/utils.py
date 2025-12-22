# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import gc
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import megatron.core
import torch
from accelerate.utils import gather as hf_gather
from accelerate.utils import gather_object as hf_gather_object
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_batch_on_this_cp_rank as mcore_get_batch_on_this_cp_rank
from megatron.training import get_args, get_wandb_writer
from packaging import version

from swift.llm import get_packed_seq_params as _get_packed_seq_params
from swift.llm import to_device
from swift.utils import get_logger
from swift.utils.torch_utils import empty_cache, get_current_device

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


def get_swift_datasets_provider(train_dataset, val_dataset):

    def swift_datasets_provider(train_val_test_num_samples, vp_stage=None):
        nonlocal val_dataset
        args = get_args()
        data_parallel_size = mpu.get_data_parallel_world_size()
        step_batch_size = args.micro_batch_size * data_parallel_size
        # To avoid errors caused by the validation set being insufficient to complete a single step.
        if val_dataset is not None and hasattr(val_dataset, '__len__') and len(val_dataset) < step_batch_size:
            val_dataset = None
        if val_dataset is None:
            args.eval_iters = 0
        return train_dataset, val_dataset, None

    return swift_datasets_provider


# Code borrowed from NVIDIA/Megatron-LM
def get_batch_on_this_tp_rank(data, vp_stage=None):
    args = get_args()

    if args.task_type == 'causal_lm':
        data['labels'] = torch.roll(data['labels'], -1, dims=-1)
        if 'loss_scale' in data:
            data['loss_scale'] = torch.roll(data['loss_scale'], -1, dims=-1)
    batch = to_device(data, 'cuda', non_blocking=True)
    if args.pipeline_model_parallel_size == 1:
        return batch
    if mcore_013:
        is_pp_first_stage = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        is_pp_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    else:
        is_pp_first_stage = mpu.is_pipeline_first_stage()
        is_pp_last_stage = mpu.is_pipeline_last_stage()
    if not args.mtp_num_layers and not is_pp_first_stage:
        batch['input_ids'] = None
    if not is_pp_last_stage:
        batch['labels'] = None
        batch['loss_scale'] = None

    return batch


def get_packed_seq_params(position_ids: torch.Tensor) -> PackedSeqParams:
    params = _get_packed_seq_params(position_ids)
    return PackedSeqParams(
        cu_seqlens_q=params['cu_seq_lens_q'],
        cu_seqlens_kv=params['cu_seq_lens_k'],
        max_seqlen_q=params['max_length_q'],
        max_seqlen_kv=params['max_length_k'],
        qkv_format='thd')


def split_cp_inputs(inputs: torch.Tensor, cu_seqlens: torch.Tensor, dim: int):
    # TODO: compat bshd
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    new_inputs = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    for i in range(cu_seqlens.shape[0] - 1):
        slices = [slice(None)] * inputs.ndim
        slices[dim] = slice(cu_seqlens[i], cu_seqlens[i + 1])
        val = inputs[tuple(slices)]
        view_shape = (*inputs.shape[:dim], 2 * cp_size, val.shape[dim] // (2 * cp_size), *inputs.shape[dim + 1:])
        val = val.view(view_shape)
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                             pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(dim, index)
        view_shape = (*inputs.shape[:dim], -1, *inputs.shape[dim + 1:])
        new_inputs.append(val.view(view_shape))
    return torch.cat(new_inputs, dim=dim)


def get_batch_on_this_cp_rank(batch: Dict[str, Any]):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        args = get_args()
        keys = ['labels', 'attention_mask', 'position_ids', 'loss_scale']
        if not args.is_multimodal:
            # Multimodal models will handle CP in input_embeds.
            keys.append('input_ids')

        packed_seq_params = batch.get('packed_seq_params')
        if packed_seq_params is None:
            return mcore_get_batch_on_this_cp_rank(batch)
        for key, val in batch.items():
            if key not in keys:
                continue
            if args.task_type == 'seq_cls' and key == 'labels':
                continue
            if val is not None:
                batch[key] = split_cp_inputs(val, packed_seq_params.cu_seqlens_q, -1)

    return batch


@contextmanager
def profiling_context(trainer, name: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    profiling_metrics = {f'profiling/Time taken: {trainer.__class__.__name__}.{name}': duration}
    wandb_writer = get_wandb_writer()
    if wandb_writer and trainer.is_main_process:
        step = getattr(getattr(wandb_writer, 'run', None), 'step', None)
        wandb_writer.log(profiling_metrics, step=step)


def profiling_decorator(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


def gather(tensor, group: Optional[torch.distributed.ProcessGroup] = None):
    if group is None:
        return hf_gather(tensor)
    size = torch.distributed.get_world_size(group=group)
    output = [torch.empty_like(tensor) for _ in range(size)]
    torch.distributed.all_gather(output, tensor, group=group, async_op=False)

    return torch.cat(output, dim=0)


def gather_object(object: Any, group: Optional[torch.distributed.ProcessGroup] = None):
    if group is None:
        return hf_gather_object(object)
    size = torch.distributed.get_world_size(group=group)
    output_objects = [None for _ in range(size)]
    torch.distributed.all_gather_object(output_objects, object, group=group)
    return [x for y in output_objects for x in y]


# code borrowed from verl
@torch.no_grad()
def load_megatron_model_to_gpu(models, load_grad=True):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # sometimes, we don't want to load grad for pure inference
                    if load_grad:
                        buffer.grad_data.storage().resize_(buffer.grad_data_size)
                        buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        # copy data from cpu to cuda
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)
        else:
            # we need this for ref module
            device_id = get_current_device()
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to(device_id, non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to(device_id, non_blocking=True)
    gc.collect()
    empty_cache()


@torch.no_grad()
def offload_megatron_model_to_cpu(models):
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group
    """
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # offload parameters
                    if buffer.param_data.storage().size() > 0:
                        buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                        buffer.param_data_size = buffer.param_data.storage().size()
                        buffer.param_data.storage().resize_(0)

                    assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

                    if buffer.grad_data.storage().size() > 0:
                        # if the grad_data size is already zero, we assume that it is already offloaded
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)
        else:
            # we need this for ref module
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to('cpu', non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to('cpu', non_blocking=True)
    gc.collect()
    empty_cache()


@torch.no_grad()
def load_megatron_copy_params(optimizers):
    """
    Load optimizer parameters back to GPU. Handles ChainedOptimizer.

    Args:
        optimizers: Optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def load_tensor_to_gpu(tensor):
        if tensor is None:
            return
        device_id = get_current_device()
        tensor.data = tensor.data.to(device_id, non_blocking=True)

    def load_group_to_gpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        load_tensor_to_gpu(param)
                else:
                    load_tensor_to_gpu(param_group)
        else:
            load_tensor_to_gpu(group)

    # Load all parameter groups to GPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, 'shard_fp32_from_float16_groups'):
            load_group_to_gpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def offload_megatron_copy_params(optimizers):
    """
    Offload optimizer parameters to CPU. Supports both Megatron optimizers
    and `ChainedOptimizer`, which wraps a list of underlying optimizers.

    Args:
        optimizers: The optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def offload_tensor_to_cpu(tensor):
        if tensor is None:
            return
        tensor.data = tensor.data.to('cpu', non_blocking=True)

    def offload_group_to_cpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        offload_tensor_to_cpu(param)
                else:
                    offload_tensor_to_cpu(param_group)
        else:
            offload_tensor_to_cpu(group)

    # Offload all parameter groups to CPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, 'shard_fp32_from_float16_groups'):
            offload_group_to_cpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def load_megatron_optimizer(optimizers):

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        load_megatron_copy_params(_opt)
        # if we are using HybridDeviceOptimizer, we need to only move gpu optimizer state to gpu
        if hasattr(_opt.optimizer, '_move_new_state_to_right_device'):
            _opt.optimizer._move_new_state_to_right_device()
        else:
            opt_state_dict_values = _opt.optimizer.state.values()
            for v in opt_state_dict_values:
                if 'exp_avg' in v:
                    v['exp_avg'] = v['exp_avg'].to(get_current_device(), non_blocking=True)
                if 'exp_avg_sq' in v:
                    v['exp_avg_sq'] = v['exp_avg_sq'].to(get_current_device(), non_blocking=True)
        gc.collect()
        empty_cache()


@torch.no_grad()
def offload_megatron_optimizer(optimizers):

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        offload_megatron_copy_params(_opt)
        opt_state_dict_values = _opt.optimizer.state.values()
        for v in opt_state_dict_values:
            if 'exp_avg' in v:
                v['exp_avg'] = v['exp_avg'].to('cpu', non_blocking=True)
            if 'exp_avg_sq' in v:
                v['exp_avg_sq'] = v['exp_avg_sq'].to('cpu', non_blocking=True)
        gc.collect()
        empty_cache()


def log_gpu_memory(prefix: str = '', info_once: bool = False):
    logger = get_logger()
    log_msg = (f'{prefix} GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, '
               f'{torch.cuda.memory_reserved()/1024**3:.2f}GB reserved')
    if info_once:
        logger.info_once(log_msg, hash_id=prefix)
    else:
        logger.info(log_msg)

# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
from accelerate.utils import gather as hf_gather
from accelerate.utils import gather_object as hf_gather_object
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_batch_on_this_cp_rank as mcore_get_batch_on_this_cp_rank
from megatron.training import get_args, get_wandb_writer

from swift.llm import get_packed_seq_params as _get_packed_seq_params
from swift.llm import to_device


def get_swift_datasets_provider(train_dataset, val_dataset):

    def swift_datasets_provider(train_val_test_num_samples):
        nonlocal val_dataset
        args = get_args()
        data_parallel_size = mpu.get_data_parallel_world_size()
        step_batch_size = args.micro_batch_size * data_parallel_size
        # To avoid errors caused by the validation set being insufficient to complete a single step.
        if val_dataset is not None and hasattr(val_dataset, '__len__') and len(val_dataset) < step_batch_size:
            val_dataset = None
        return train_dataset, val_dataset, None

    return swift_datasets_provider


# Code borrowed from NVIDIA/Megatron-LM
def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()

    data = next(data_iterator)
    is_finished = data.pop('is_finished', False)
    if args.task_type == 'causal_lm':
        data['labels'] = torch.roll(data['labels'], -1, dims=-1)
        if 'loss_scale' in data:
            data['loss_scale'] = torch.roll(data['loss_scale'], -1, dims=-1)
    batch = to_device(data, 'cuda', non_blocking=True)
    if args.pipeline_model_parallel_size == 1:
        pass
    elif mpu.is_pipeline_first_stage():
        batch['labels'] = None
        batch['loss_scale'] = None
    elif mpu.is_pipeline_last_stage():
        batch['input_ids'] = None
    else:
        for key in ('input_ids', 'labels', 'loss_scale'):
            batch[key] = None

    if is_finished:
        args.train_iters = args.curr_iteration + 1

    return batch


def get_packed_seq_params(position_ids: torch.Tensor) -> PackedSeqParams:
    params = _get_packed_seq_params(position_ids)
    return PackedSeqParams(
        cu_seqlens_q=params['cu_seq_lens_q'],
        cu_seqlens_kv=params['cu_seq_lens_k'],
        max_seqlen_q=params['max_length_q'],
        max_seqlen_kv=params['max_length_k'],
        qkv_format='thd')


def process_packed_seq_params(batch: Dict[str, Any]) -> int:
    args = get_args()
    num_samples = batch.pop('num_samples')
    text_position_ids = batch.pop('text_position_ids', None)
    if text_position_ids is None:
        text_position_ids = batch.get('position_ids')
    if args.padding_free and text_position_ids is not None:
        batch['packed_seq_params'] = get_packed_seq_params(text_position_ids)
        batch['packed_seq_params'].num_samples = num_samples
    return batch


def split_cp_inputs(inputs: torch.Tensor, cu_seqlens: torch.Tensor, dim: int):
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    new_inputs = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    for i in range(cu_seqlens.shape[0] - 1):
        slices = [slice(None)] * inputs.ndim
        slices[dim] = slice(cu_seqlens[i], cu_seqlens[i + 1])
        val = inputs[slices]
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
        if args.model_meta.is_multimodal:
            keys.append('decoder_input')
        else:
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


def get_batch(data_iterator):
    """Generate a batch."""
    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    # process batch for packed sequence support
    batch = process_packed_seq_params(batch)
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
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
        wandb_writer.log(profiling_metrics)

    # TODO: add swanlab support


def gather_dict(tensors: Dict[str, torch.Tensor], group: torch.distributed.ProcessGroup):
    if not isinstance(tensors, dict):
        raise ValueError(f'Expected a dictionary, got {type(tensors)}')
    size = torch.distributed.get_world_size(group=group)

    output = {}
    sorted_keys = sorted(tensors.keys())
    for key in sorted_keys:
        val = tensors[key]
        if isinstance(val, int):
            # num_samples
            output[key] = val
            continue
        elif isinstance(val, torch.Tensor):
            output[key] = [torch.empty_like(val) for _ in range(size)]
            torch.distributed.all_gather(output[key], val, group=group, async_op=False)
            output[key] = torch.cat(output[key], dim=0)
        else:
            output[key] = [None for _ in range(size)]
            torch.distributed.all_gather_object(output[key], val, group=group, async_op=False)
            output[key] = [item for sublist in output[key] for item in sublist]

    return output


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
    torch.distributed.all_gather_object(output_objects, object)
    # all_gather_object returns a list of lists, so we need to flatten it
    return [x for y in output_objects for x in y]


def make_batch_generator(batch: List[Dict[str, Any]], batch_size: int):
    assert batch_size > 0, 'batch_size must be positive'
    assert len(batch) % batch_size == 0, 'batch length must be a multiple of batch_size'
    for i in range(0, len(batch), batch_size):
        yield batch[i:i + batch_size]

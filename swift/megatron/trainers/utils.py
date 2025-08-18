# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_batch_on_this_cp_rank as mcore_get_batch_on_this_cp_rank
from megatron.training import get_args

from swift.llm import get_packed_seq_params as _get_packed_seq_params


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

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        data = next(data_iterator)
        is_finished = data.pop('is_finished', False)
        input_ids = data['input_ids']
        seq_length = input_ids.shape[1]
        has_loss_scale = 'loss_scale' in data
        data['labels'] = torch.roll(data['labels'], -1, dims=-1)
        if has_loss_scale:
            data['loss_scale'] = torch.roll(data['loss_scale'], -1, dims=-1)
        batch = {
            'input_ids': input_ids.cuda(non_blocking=True),
            'labels': data['labels'].cuda(non_blocking=True),
            'attention_mask': None if 'attention_mask' not in data else data['attention_mask'].cuda(non_blocking=True),
            'position_ids': data['position_ids'].cuda(non_blocking=True),
            'loss_scale': None if not has_loss_scale else data['loss_scale'].cuda(non_blocking=True),
            'num_samples': data['num_samples'],
        }
        flags = torch.tensor([seq_length, is_finished, has_loss_scale, data['num_samples']]).cuda(non_blocking=True)
        _broadcast(flags)
        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['input_ids'])
            _broadcast(batch['labels'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast(batch['loss_scale'])

        elif mpu.is_pipeline_first_stage():
            batch['labels'] = None
            batch['loss_scale'] = None
            _broadcast(batch['input_ids'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            batch['input_ids'] = None
            _broadcast(batch['labels'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast(batch['loss_scale'])
        else:
            for key in ('input_ids', 'labels', 'attention_mask', 'position_ids', 'loss_scale'):
                batch[key] = None

    else:
        flags = torch.empty((4), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(flags)
        seq_length, is_finished, has_loss_scale, num_samples = flags.tolist()
        if args.padding_free:
            micro_batch_size = 1  # use qkv_format 'thd'
            attention_mask = None
        else:
            micro_batch_size = args.micro_batch_size
            attention_mask = torch.empty((micro_batch_size, 1, seq_length, seq_length),
                                         dtype=torch.bool,
                                         device=torch.cuda.current_device())
        input_ids = torch.empty((micro_batch_size, seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty((micro_batch_size, seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        loss_scale = torch.empty(
            (micro_batch_size,
             seq_length), dtype=torch.float32, device=torch.cuda.current_device()) if has_loss_scale else None
        position_ids = torch.empty((micro_batch_size, seq_length),
                                   dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(input_ids)
            _broadcast(labels)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            _broadcast(loss_scale)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_scale = None

            _broadcast(input_ids)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            input_ids = None

            _broadcast(labels)
            _broadcast(attention_mask)
            _broadcast(position_ids)  # compat packing & cp
            _broadcast(loss_scale)
        else:
            input_ids, labels, attention_mask, position_ids, loss_scale = (None, ) * 5

        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_scale': loss_scale,
            'num_samples': num_samples,
        }
    if is_finished:
        args.train_iters = args.curr_iteration + 1

    return batch


def get_packed_seq_params(position_ids: torch.Tensor) -> PackedSeqParams:
    params = _get_packed_seq_params(position_ids)
    return PackedSeqParams(
        cu_seqlens_q=params['cumulative_seqlens_q'],
        cu_seqlens_kv=params['cumulative_seqlens_k'],
        max_seqlen_q=params['max_length_q'],
        max_seqlen_kv=params['max_length_k'],
        qkv_format='thd')


def _split_tokens(tokens, cu_seqlens):
    assert tokens.shape[0] == 1, f'tokens.shape: {tokens.shape}'
    new_tokens = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    for i in range(cu_seqlens.shape[0] - 1):
        val = tokens[:, cu_seqlens[i]:cu_seqlens[i + 1]]
        val = val.view(
            tokens.shape[0],
            2 * cp_size,
            val.shape[1] // (2 * cp_size),
        )
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                             pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(1, index)
        new_tokens.append(val.view(tokens.shape[0], -1))
    return torch.cat(new_tokens, dim=1)


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
        packed_seq_params = batch.get('packed_seq_params')
        if packed_seq_params is None:
            return mcore_get_batch_on_this_cp_rank(batch)
        for key, val in batch.items():
            if key == 'packed_seq_params':
                continue
            if val is not None:
                batch[key] = _split_tokens(val, packed_seq_params.cu_seqlens_q)

    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    args = get_args()
    num_samples = batch.pop('num_samples')
    if args.padding_free and batch.get('position_ids') is not None:
        batch['packed_seq_params'] = get_packed_seq_params(batch['position_ids'])
        batch['packed_seq_params'].num_samples = num_samples
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    return batch

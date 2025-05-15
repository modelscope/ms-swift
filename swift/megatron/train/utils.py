# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Any, Dict, Optional

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers
from megatron.training.training import cyclic_iter

from swift.llm import DataLoaderDispatcher

stimer = StragglerDetector()


def get_swift_datasets_provider(train_dataset, val_dataset):

    def swift_datasets_provider(train_val_test_num_samples):
        return train_dataset, val_dataset, None

    return swift_datasets_provider


class MegatronDataLoaderDispatcher(DataLoaderDispatcher):

    @property
    def group(self):
        return mpu.get_data_parallel_group()


def build_streaming_dataloader(args, dataset, collate_fn):
    base_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=args.micro_batch_size,
        prefetch_factor=args.dataloader_prefetch_factor,
        persistent_workers=args.dataloader_persistent_workers,
    )
    return iter(cyclic_iter(MegatronDataLoaderDispatcher(base_dataloader)))


def get_batch_on_this_tp_rank(data_iterator):
    # copy from megatron-lm

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        try:
            data = next(data_iterator)
        except StopIteration:
            seq_length = -1
        else:
            tokens = data['input_ids']
            seq_length = tokens.shape[1]
            batch = {
                'tokens': tokens.cuda(non_blocking=True),
                'labels': data['labels'].cuda(non_blocking=True),
                'attention_mask':
                None if 'attention_mask' not in data else data['attention_mask'].cuda(non_blocking=True),
                'position_ids': data['position_ids'].cuda(non_blocking=True)
            }
        seq_length = torch.tensor(seq_length).cuda(non_blocking=True)
        _broadcast(seq_length)
        if seq_length.item() == -1:
            return {}
        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

    else:
        seq_length = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(seq_length)
        if seq_length.item() == -1:
            return {}
        micro_batch_size = 1  # use qkv_format 'thd'
        tokens = torch.empty((micro_batch_size, seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty((micro_batch_size, seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty((micro_batch_size, 1, seq_length, seq_length),
                                         dtype=torch.bool,
                                         device=torch.cuda.current_device())
        else:
            attention_mask = None
        position_ids = torch.empty((micro_batch_size, seq_length),
                                   dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None

            _broadcast(labels)
            _broadcast(attention_mask)
            _broadcast(position_ids)  # compat packing & cp

        batch = {'tokens': tokens, 'labels': labels, 'attention_mask': attention_mask, 'position_ids': position_ids}

    return batch


def get_packed_seq_params(position_ids: torch.Tensor) -> Optional[PackedSeqParams]:
    position_ids_f = position_ids.flatten()
    indices_q = torch.arange(position_ids_f.shape[0], device=position_ids_f.device, dtype=torch.int32)

    cu_seqlens = torch.cat([
        indices_q[position_ids_f == 0],
        torch.tensor(position_ids_f.shape, device=position_ids_f.device, dtype=torch.int32),
    ])

    max_length = position_ids_f.max() + 1
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_length,
        max_seqlen_kv=max_length,
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
        packed_seq_params = batch['packed_seq_params']
        for key, val in batch.items():
            if key == 'packed_seq_params':
                continue
            if val is not None:
                batch[key] = _split_tokens(val, packed_seq_params.cu_seqlens_q)

    return batch


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    if not batch:
        return batch
    batch['packed_seq_params'] = get_packed_seq_params(batch['position_ids'])
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


def forward_step(data_iterator, model):
    from pretrain_gpt import loss_func

    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        data = get_batch(data_iterator)
    if not data:
        raise StopIteration
    tokens, labels, attention_mask, position_ids, packed_seq_params = data
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)
    loss_mask = None if labels is None else (labels != -100).float()
    return output_tensor, partial(loss_func, loss_mask)

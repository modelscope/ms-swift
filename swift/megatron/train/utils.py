# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Optional

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
    def src_rank(self):
        return mpu.get_data_parallel_src_rank()

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
        prefetch_factor=10,
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

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        tokens = data['input_ids']
        seq_length = torch.tensor(tokens.shape[1]).cuda(non_blocking=True)
        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': data['labels'].cuda(non_blocking=True),
            'attention_mask': None if 'attention_mask' not in data else data['attention_mask'].cuda(non_blocking=True),
            'position_ids': data['position_ids'].cuda(non_blocking=True)
        }
        _broadcast(seq_length)
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

    else:
        seq_length = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(seq_length)

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
            position_ids = None

            _broadcast(labels)
            _broadcast(attention_mask)

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


def forward_step(data_iterator, model):
    import pretrain_gpt
    from pretrain_gpt import loss_func, get_batch
    # patch get_batch_on_this_tp_rank
    pretrain_gpt.get_batch_on_this_tp_rank = get_batch_on_this_tp_rank

    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, attention_mask, position_ids = get_batch(data_iterator)
        packed_seq_params = None if position_ids is None else get_packed_seq_params(position_ids)

    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)
    loss_mask = None if labels is None else (labels != -100).float()
    return output_tensor, partial(loss_func, loss_mask)

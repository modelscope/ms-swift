# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.core import mpu
from megatron.training import get_args


def get_swift_datasets_provider(train_dataset, val_dataset):

    def swift_datasets_provider(train_val_test_num_samples):
        return train_dataset, val_dataset, None

    return swift_datasets_provider


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
        seq_length = torch.tensor(data['tokens'].shape[1]).cuda(non_blocking=True)
        batch = {
            'tokens': data['tokens'].cuda(non_blocking=True),
            'labels': data['labels'].cuda(non_blocking=True),
            'loss_mask': data['loss_mask'].cuda(non_blocking=True),
            'attention_mask': None if 'attention_mask' not in data else data['attention_mask'].cuda(non_blocking=True),
            'position_ids': data['position_ids'].cuda(non_blocking=True)
        }
        _broadcast(seq_length)
        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:
        seq_length = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(seq_length)

        tokens = torch.empty((args.micro_batch_size, seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, seq_length), dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, seq_length),
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty((args.micro_batch_size, 1, seq_length, seq_length),
                                         dtype=torch.bool,
                                         device=torch.cuda.current_device())
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, seq_length),
                                   dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch

from functools import partial

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.training import get_args
from megatron.training.utils import average_losses_across_data_parallel_group, get_batch_on_this_cp_rank


def get_batch_on_this_tp_rank(data_iterator):
    # copy from Megatron-LM and made some changes.
    args = get_args()

    def _broadcast(item):
        if item is not None:
            dist.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        args.seq_length = data['tokens'].shape[1]
        _broadcast(torch.tensor(args.seq_length).cuda(non_blocking=True))
        batch = {
            'tokens': data['tokens'].cuda(non_blocking=True),
            'labels': data['labels'].cuda(non_blocking=True),
            'loss_mask': data['loss_mask'].cuda(non_blocking=True),
            'attention_mask': None if 'attention_mask' not in data else data['attention_mask'].cuda(non_blocking=True),
            'position_ids': data['position_ids'].cuda(non_blocking=True)
        }

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
        args.seq_length = seq_length.item()
        tokens = torch.empty((args.micro_batch_size, args.seq_length),
                             dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length),
                             dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length),
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty((args.micro_batch_size, 1, args.seq_length, args.seq_length),
                                         dtype=torch.bool,
                                         device=torch.cuda.current_device())
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, args.seq_length),
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


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function. copy from Pai-Megatron-Patch

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """

    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        dist.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = dist.get_rank()
        assert not loss.isnan(), (f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                  f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)

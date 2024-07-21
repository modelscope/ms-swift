import os
import sys
from dataclasses import asdict, dataclass, field
from functools import partial, wraps
from typing import Any, Dict, List, Optional, Tuple

import torch
from megatron.core import mpu
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from megatron.training.utils import (average_losses_across_data_parallel_group, get_batch_on_this_cp_rank,
                                     get_batch_on_this_tp_rank, get_ltor_masks_and_position_ids)
from megatron_patch.model.qwen2.model import GPTModel

from swift.utils import get_dist_setting, get_logger, subprocess_run
from .dataset import get_dataset
from .template import get_template
from .utils import LazyLLMDataset, to_device

logger = get_logger()


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
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                  f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel):
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    # train_val_test_num_samples: ignored
    args = get_args()
    tokenizer = get_tokenizer()
    logger.info('> building train, validation, and test datasets for GPT ...')
    # Loading Dataset
    train_dataset, val_dataset = get_dataset(args.dataset, 0.01)
    template = get_template(args.template_type, tokenizer)
    train_dataset = LazyLLMDataset(train_dataset, template)
    if val_dataset is not None:
        val_dataset = LazyLLMDataset(val_dataset, template)

    from megatron.training import training
    assert not hasattr(training, '_old_build_pretraining_data_loader')
    _old_build_pretraining_data_loader = training.build_pretraining_data_loader

    def data_collator(batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = template.data_collator(batch, padding_to)
        labels = res['labels']
        new_labels = torch.zeros_like(labels)
        new_labels[:, :-1] = labels[:, 1:]
        new_labels[:, -1] = -100
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(new_labels, -100, False, False, True)
        return {
            'tokens': res['input_ids'],
            'labels': new_labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }

    @wraps(_old_build_pretraining_data_loader)
    def build_pretraining_data_loader(*args, **kwargs):
        res = _old_build_pretraining_data_loader(*args, **kwargs)
        if res is not None:
            res.collate_fn = data_collator
        return res

    training.build_pretraining_data_loader = build_pretraining_data_loader
    training._old_build_pretraining_data_loader = _old_build_pretraining_data_loader
    return train_dataset, val_dataset, None

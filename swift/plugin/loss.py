# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Callable, Optional

import torch
from torch.nn import CrossEntropyLoss


class LossType:
    loss_scale = 'loss_scale'


LOSS_MAPPING = {}


def register_loss_func(loss_type: str, loss_func: Optional[Callable] = None):
    loss_info = {}

    if loss_func is not None:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_type] = loss_info
        return

    def _register_loss_func(loss_func: Callable) -> Callable:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_type] = loss_info
        return loss_func

    return _register_loss_func


def ce_loss_func(outputs, labels):
    logits = outputs.logits
    device = logits.device
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:].to(device)
    # Save memory
    masks = shift_labels != -100
    shift_logits = shift_logits[masks]
    shift_labels = shift_labels[masks]
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits, shift_labels)
    return loss, masks


# Use @register_loss_func to decorate your own loss, use --loss_type xxx to train
@register_loss_func(LossType.loss_scale)
def loss_scale_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    """Loss func

    Args:
        outputs: The model outputs
        labels: The labels
        loss_scale: The loss scale
        num_items_in_batch: Number of tokens in the labels of gradient accumulation round that are not -100.

    Returns:

    """
    loss, masks = ce_loss_func(outputs, labels)
    if loss_scale is not None:
        shift_scale = loss_scale[..., 1:].to(masks.device)
        shift_scale = shift_scale[masks]
        loss = (shift_scale * loss)
    if num_items_in_batch is None:
        loss = loss.mean()
    else:
        # compat transformers>=4.46
        loss = loss.sum() / num_items_in_batch
    return loss


def get_loss_func(loss_type: Optional[str]) -> Optional[Callable]:
    if loss_type is None:
        return None
    return LOSS_MAPPING[loss_type]['loss_func']

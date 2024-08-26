from typing import Callable, Optional

import torch
from torch.nn import CrossEntropyLoss


class LossName:
    long_ce = 'long-ce'
    loss_scale = 'loss-scale'


LOSS_MAPPING = {}


def register_loss_func(loss_name: str, loss_func: Optional[Callable] = None):
    loss_info = {}

    if loss_func is not None:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_name] = loss_info
        return

    def _register_loss_func(loss_func: Callable) -> Callable:
        loss_info['loss_func'] = loss_func
        LOSS_MAPPING[loss_name] = loss_info
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
    return loss


class LongCrossEntropy:
    """Assign higher weight to long text."""

    def __init__(self, length_smooth: float = 0.9):
        self._s_length = 0
        self._norm_factor = 0
        self._smoothing = length_smooth

    def __call__(self, outputs, labels) -> torch.Tensor:
        # moving average
        loss = ce_loss_func(outputs, labels)
        self._s_length = self._s_length * self._smoothing + loss.shape[0]
        self._norm_factor = self._norm_factor * self._smoothing + 1
        loss = loss.sum() / (self._s_length / self._norm_factor)
        return loss


register_loss_func(LossName.long_ce, LongCE())


@register_loss_func(LossName.loss_scale)
def loss_scale_func(outputs, labels, loss_scale=None) -> torch.Tensor:
    loss = ce_loss_func(outputs, labels)
    if loss_scale is None:
        loss = loss.mean()
    else:
        shift_scale = loss_scale[..., 1:].to(device)
        shift_scale = shift_scale[masks]
        loss = (shift_scale * loss).mean()
    return loss


def get_loss_func(loss_name: str) -> Optional[Callable]:
    if loss_name.lower() == 'auto':
        return None
    return LOSS_MAPPING[loss_name]['loss_func']

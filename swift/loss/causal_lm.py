# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import BaseLoss


class CustomCrossEntropyLoss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        from swift.trainers import per_token_loss_func
        token_loss = per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        return token_loss.sum() / num_items_in_batch

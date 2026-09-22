# Copyright (c) ModelScope Contributors. All rights reserved.
import torch.distributed as dist

from .base import BaseLoss


class CustomCrossEntropyLoss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        if self.trainer is not None and self.trainer.template.sequence_parallel_size > 1:
            # The trainer already shifted, gathered and weighted the per-token loss.
            loss = outputs.loss.sum()
            if num_items_in_batch is None:
                num_items_in_batch = (labels != -100).sum()
                dist.all_reduce(num_items_in_batch, op=dist.ReduceOp.SUM)
                # Counts inferred inside this callback are not visible to the trainer's rescaling step.
                if (getattr(self.trainer.args, 'average_tokens_across_devices', False)
                        and self.trainer.model_accepts_loss_kwargs):
                    loss = loss * self.trainer.accelerator.num_processes
                    if not self.trainer.model.training:
                        loss = loss / self.trainer.template.sequence_parallel_size
            return loss / num_items_in_batch

        from swift.trainers import per_token_loss_func
        token_loss = per_token_loss_func(outputs, labels)
        if loss_scale is not None:
            token_loss = token_loss * loss_scale
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        return token_loss.sum() / num_items_in_batch

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Abstract base for worker-side loss computation.

A ``Loss`` subclass defines ``forward_step`` and ``loss_func`` -- the two
methods that Megatron's pipeline-parallel scheduler calls during
training.  Users who want to customise loss computation only need to
subclass ``Loss`` and override these two methods; no understanding of
the internal Megatron trainer is required.

Example::

    class MyLoss(Loss):
        def __init__(self, args):
            self.label_smoothing = args.label_smoothing

        def forward_step(self, data_iterator, model):
            batch = next(data_iterator)
            output = model(batch['input_ids'], ...)
            return output, partial(self.loss_func, labels=batch['labels'])

        def loss_func(self, output_tensor, *, labels):
            loss = F.cross_entropy(output_tensor, labels)
            return loss, {'loss': loss.item()}

Then register it::

    register_ray_trainer('my_algo', trainer='...MyDriver', loss='...MyLoss')
"""
from abc import ABC, abstractmethod


class Loss(ABC):
    """Abstract base for worker-side loss / forward computation.

    Mirrors the two methods that Megatron's PP scheduler calls:
    ``forward_step(data_iterator, model) -> (output, loss_fn)``
    and ``loss_func(output_tensor, **ctx) -> (loss, metrics)``.

    Subclasses may wrap an existing trainer via composition for code
    reuse (see ``GRPOLoss``) or implement these from scratch.
    """

    @abstractmethod
    def forward_step(self, data_iterator, model):
        """Run a single forward micro-batch through *model*.

        Returns ``(output_tensor, partial(self.loss_func, ...))``.
        """

    @abstractmethod
    def loss_func(self, output_tensor, **kwargs):
        """Compute scalar loss + metrics from ``output_tensor``.

        Returns ``(loss, metric_dict)``.
        """

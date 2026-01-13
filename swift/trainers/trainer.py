# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
from contextlib import contextmanager
from functools import wraps

import torch
from peft import PeftModel
from transformers import Trainer as HfTrainer

from swift.sequence_parallel import sequence_parallel
from swift.utils import get_logger
from .arguments import TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin

logger = get_logger()


class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    args: TrainingArguments

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        # For tasks whose `labels` are per-sample (e.g. seq_cls/reranker/embedding), we must NOT let
        # SP code treat them as token labels. We detect that case by `labels.dim() == 1` and temporarily
        # remove labels during `prepare_inputs`.
        if self.template.sequence_parallel_size > 1:
            labels = inputs.get('labels', None)
            pop_labels = isinstance(labels, torch.Tensor) and labels.dim() == 1
            if pop_labels:
                labels = inputs.pop('labels', None)
            try:
                sequence_parallel.prepare_inputs(inputs)
            finally:
                if pop_labels and labels is not None:
                    inputs['labels'] = labels
        return inputs

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss

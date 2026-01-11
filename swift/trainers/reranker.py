# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict

from transformers import EvalPrediction

from swift.utils import get_generative_reranker_logits, get_logger
from .trainer import Trainer
from .utils import gather_for_unpadded_tensors

logger = get_logger()


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if inputs.get('attention_mask') is None and self.template.padding_side != 'left':
            raise ValueError('When using padding_free, padding_side must be set to "left".')
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            # Get labels and compute outputs
            labels = inputs.pop('labels', None)
            outputs = model(**inputs)
            if self.args.task_type == 'generative_reranker':
                outputs.logits = get_generative_reranker_logits(
                    self.tokenizer, outputs.logits, attention_mask=inputs.get('attention_mask'))

            if labels is not None:
                # Call custom loss function
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            else:
                # Fallback to model's loss
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

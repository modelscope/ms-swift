# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from swift.utils import get_last_valid_indices, get_logger
from .trainer import Trainer
from .utils import gather_for_unpadded_tensors

logger = get_logger()


def gather_for_reranker_metrics(input_data, use_gather_object=False):
    if isinstance(input_data, tuple):
        return tuple(gather_for_reranker_metrics(data, use_gather_object=use_gather_object) for data in input_data)
    if isinstance(input_data, list):
        return [gather_for_reranker_metrics(data, use_gather_object=use_gather_object) for data in input_data]
    return gather_for_unpadded_tensors(input_data, use_gather_object=use_gather_object)


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather_function = gather_for_reranker_metrics
        if getattr(self.args, 'loss_type', None) == 'pointwise_reranker' and 'group_sizes' not in self.label_names:
            self.label_names.append('group_sizes')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            # Get labels and compute outputs
            labels = inputs.pop('labels', None)
            group_sizes = inputs.pop('group_sizes', None)
            outputs = model(**inputs)
            if self.task_type == 'generative_reranker':
                logits = outputs.logits
                attention_mask = inputs.get('attention_mask')
                last_valid_indices = -1 if attention_mask is None else get_last_valid_indices(attention_mask)
                batch_indices = torch.arange(logits.shape[0], device=logits.device)
                outputs.logits = logits[batch_indices, last_valid_indices]

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

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_reranker_metrics
        return output

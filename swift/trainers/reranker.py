# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict

from transformers import EvalPrediction

from swift.utils import get_logger
from .trainer import Trainer
from .utils import gather_for_unpadded_tensors

logger = get_logger()


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.include_for_metrics = ['inputs']
        self.compute_metrics = self.calculate_metric
        self.label_names = ['labels']

        # Set up preprocess_logits_for_metrics to reduce memory usage for generative reranker
        if self.args.loss_type in {'generative_reranker', 'listwise_generative_reranker'}:
            self.preprocess_logits_for_metrics = self._preprocess_generative_reranker_logits
        else:
            self.preprocess_logits_for_metrics = None
        self.gather_function = gather_for_unpadded_tensors

    def _preprocess_generative_reranker_logits(self, logits, labels):
        """
        Preprocess logits for generative reranker to reduce memory usage.
        Extract only the yes/no token logits at the last valid (non -100) timestep
        for each sample, avoiding padded timesteps created by multi-GPU gather.
        """

        # Get token IDs for positive and negative tokens
        positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
        negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

        tokenizer = getattr(self, 'processing_class', None)
        if tokenizer is None:
            # Fallback: return full logits if tokenizer not available
            return logits

        try:
            positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
            negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
        except Exception:
            # Fallback: return full logits if token conversion fails
            return logits

        # Extract only the yes/no token logits from the last non -100 position per sample
        # Shapes: logits [batch, seq_len, vocab]
        if len(logits.shape) == 3:
            positive_logits = logits[:, :, positive_token_id]
            negative_logits = logits[:, :, negative_token_id]
            logits = positive_logits - negative_logits
            return logits
        else:
            # Unexpected shape, return as-is
            return logits

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        import numpy as np
        input_ids = eval_prediction.inputs
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids

        if self.template.padding_free:
            logits = logits[:, -1]
        else:
            if logits.ndim == 2 and logits.shape[1] > 1:
                pad_token_id = self.tokenizer.pad_token_id
                valid_mask = (input_ids != pad_token_id) & (input_ids != -100)
                last_valid_indices = valid_mask[:, ::-1].argmax(axis=1)
                last_valid_indices = input_ids.shape[1] - 1 - last_valid_indices
                logits = logits[np.arange(logits.shape[0]), last_valid_indices]
        return calculate_reranker_metrics(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if inputs.get('attention_mask') is None and self.template.padding_side != 'left':
            raise ValueError('When using padding_free, padding_side must be set to "left".')
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            # Get labels and compute outputs
            labels = inputs.get('labels')
            if labels is not None:
                labels = inputs.pop('labels')

            outputs = model(**inputs)

            if labels is not None:
                # Call custom loss function
                loss = self.compute_loss_func(
                    outputs,
                    labels,
                    num_items_in_batch=num_items_in_batch,
                    trainer=self,
                    attention_mask=inputs.get('attention_mask'))
            else:
                # Fallback to model's loss
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels, attention_mask=inputs.get('attention_mask'))

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from swift.metrics.base import EvalMetrics
from swift.utils import get_logger
from .trainer import Trainer
from .utils import gather_for_unpadded_tensors

logger = get_logger()


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._supports_embedding_gather = hasattr(self, 'gather_function')
        self._gather_embedding_samples = False
        self.gather_function = gather_for_unpadded_tensors
        mrl_dims = self.args.mrl_dims
        if mrl_dims and self.compute_loss_func is not None:
            origin_loss_func = self.compute_loss_func

            def mrl_loss_func(outputs, labels, **kwargs):
                # Matryoshka Representation Learning: compute loss on each truncated dimension
                # and aggregate with the corresponding weights.
                last_hidden_state = outputs['last_hidden_state']
                loss = None
                for dim, weight in mrl_dims.items():
                    if dim > last_hidden_state.shape[-1]:
                        logger.warning_once(f'MRL: skipping dimension {dim} because it exceeds the model hidden size '
                                            f'({last_hidden_state.shape[-1]}).')
                        continue
                    sliced = F.normalize(last_hidden_state[..., :dim], p=2, dim=-1)
                    cur_loss = weight * origin_loss_func({'last_hidden_state': sliced}, labels, **kwargs)
                    loss = cur_loss if loss is None else loss + cur_loss
                return loss

            self.compute_loss_func = mrl_loss_func

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, predictions, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if self._gather_embedding_samples and predictions is not None and labels is not None:
            # Accelerate removes padded evaluation samples along dimension zero. Keep
            # each anchor/positive/negative group together until after that gather.
            if self.args.eval_metric == 'infonce':
                starts = torch.nonzero(labels, as_tuple=True)[0].tolist()
                lengths = [end - start for start, end in zip(starts, starts[1:] + [len(labels)])]
            else:
                lengths = [1] * len(labels)
            predictions = self._pack_embedding_samples(predictions, [length + 1 for length in lengths])
            labels = self._pack_embedding_samples(labels, lengths)
        return loss, predictions, labels

    @staticmethod
    def _pack_embedding_samples(values, lengths):
        groups = torch.split(values, lengths)
        masks = [torch.ones(length, device=values.device, dtype=torch.long) for length in lengths]
        return {
            'embedding_values': pad_sequence(groups, batch_first=True),
            'embedding_mask': pad_sequence(masks, batch_first=True),
        }

    def _gather_embedding_outputs(self, inputs):
        if not isinstance(inputs, dict) or 'embedding_mask' not in inputs:
            return gather_for_unpadded_tensors(inputs)
        outputs = self.accelerator.gather_for_metrics(inputs)
        # Unpack each batch immediately: the evaluation cache must not pad all
        # examples to the largest negative count in the entire dataset.
        return outputs['embedding_values'][outputs['embedding_mask'] == 1].cpu()

    def evaluation_loop(self, *args, **kwargs):
        preprocess = getattr(self.preprocess_logits_for_metrics, '__func__', None)
        self._gather_embedding_samples = (
            self._supports_embedding_gather and self.template.sequence_parallel_size == 1
            and self.args.eval_metric in {'infonce', 'paired'}
            and preprocess is EvalMetrics.preprocess_logits_for_metrics)
        self.gather_function = (
            self._gather_embedding_outputs if self._gather_embedding_samples else gather_for_unpadded_tensors)
        try:
            return super().evaluation_loop(*args, **kwargs)
        finally:
            self._gather_embedding_samples = False
            self.gather_function = gather_for_unpadded_tensors

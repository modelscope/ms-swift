# Copyright (c) ModelScope Contributors. All rights reserved.
from functools import partial

import torch.nn

from swift.loss import loss_map
from swift.metrics import eval_metrics_map
from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronEmbeddingTrainer(BaseMegatronTrainer):

    def __init__(self, args, template):
        super().__init__(args, template)
        if args.context_parallel_size > 1:
            raise ValueError('Currently `task_type="embedding"` does not support context parallelism.')
        if not args.padding_free:
            raise ValueError('Currently, task_type embedding only supports padding_free.')
        self._loss_func = loss_map[args.loss_type](args, self)
        eval_metric = 'infonce' if args.loss_type == 'infonce' else 'paired'
        self.eval_metrics = eval_metrics_map[eval_metric](args, self)

    def loss_func(self, output_tensor: torch.Tensor, *, labels: torch.Tensor, packed_seq_params=None):
        training = self.unwrapped_models[0].training
        last_hidden_state = self.get_last_tokens(output_tensor, packed_seq_params)
        if not training:
            self.eval_metrics.update(last_hidden_state.detach(), labels)
        loss = self._loss_func({'last_hidden_state': last_hidden_state}, labels)
        metric = {'loss': loss.detach().clone()}
        metric = self._all_reduce_metric(metric)
        return loss, metric

    def forward_step(self, data_iterator, model):
        # Get the batch.
        vp_stage = model.module.module.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        labels = data.pop('labels', None)
        output_tensor = model(**data)
        packed_seq_params = data.get('packed_seq_params')
        loss_func = partial(self.loss_func, labels=labels, packed_seq_params=packed_seq_params)
        return output_tensor, loss_func

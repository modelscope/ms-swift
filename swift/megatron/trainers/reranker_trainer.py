# Copyright (c) ModelScope Contributors. All rights reserved.
from collections import namedtuple
from functools import partial

import torch.nn
from megatron.training import get_args, get_timers

from swift.loss import loss_map
from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()
ModelOutputs = namedtuple('ModelOutputs', ['logits'])


class MegatronRerankerTrainer(BaseMegatronTrainer):

    def __init__(self, args, template):
        super().__init__(args, template)
        if args.context_parallel_size > 1:
            raise ValueError('Currently `task_type="reranker/generative_reranker"` does not support '
                             'context parallelism.')
        if not args.padding_free:
            raise ValueError('Currently, task_type reranker/generative_reranker only supports padding_free.')
        self._loss_func = loss_map[self.args.loss_type](args, self)

    def loss_func(self, output_tensor: torch.Tensor, *, labels: torch.Tensor, packed_seq_params=None):
        logits = self.get_last_tokens(output_tensor, packed_seq_params)
        loss = self._loss_func(ModelOutputs(logits=logits), labels)
        metric = {'loss': loss.detach().clone()}
        metric = self._all_reduce_metric(metric)
        return loss, metric

    def setup_model_and_optimizer(self, *_args, **kwargs):
        res = super().setup_model_and_optimizer(*_args, **kwargs)
        for model in self.unwrapped_models:
            model.tokenizer = self.template.tokenizer
        return res

    def forward_step(self, data_iterator, model):
        timers = get_timers()

        # Get the batch.
        vp_stage = model.module.module.vp_stage
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()
        labels = data.get('labels')
        if self.args.task_type == 'seq_cls':
            data.pop('labels', None)
        with self.stimer:
            output_tensor = model(**data)
        packed_seq_params = data.get('packed_seq_params')
        loss_func = partial(self.loss_func, labels=labels, packed_seq_params=packed_seq_params)
        return output_tensor, loss_func

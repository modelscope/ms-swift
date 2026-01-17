# Copyright (c) ModelScope Contributors. All rights reserved.
import torch.nn
from megatron.training import get_args, get_timers
from functools import partial
from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronEmbeddingTrainer(BaseMegatronTrainer):

    def loss_func(self, output_tensor: torch.Tensor, *, labels: torch.Tensor, packed_seq_params=None):
        args = self.args
        if not args.padding_free:
            raise ValueError('Currently, task_type embedding only supports padding_free.')

        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        num_samples = packed_seq_params.num_samples
        logits = cu_seqlens_q[1:num_samples + 1] - 1

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

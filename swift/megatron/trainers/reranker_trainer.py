# Copyright (c) ModelScope Contributors. All rights reserved.
import torch.nn
from megatron.training import get_args, get_timers

from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronRerankerTrainer(BaseMegatronTrainer):

    def loss_func(self, output_tensor: torch.Tensor, *, labels: torch.Tensor, packed_seq_params=None):
        pass

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

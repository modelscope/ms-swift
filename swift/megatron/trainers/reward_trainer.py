# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from functools import partial
from typing import Literal

import torch
from megatron.core import mpu
from megatron.training import get_args, get_timers
from torch import nn
from trl import KTOTrainer

from swift.utils import get_current_device, get_logger
from .rlhf_mixin import MegatronRLHFTrainer

logger = get_logger()


class MegatronRewardTrainer(MegatronRLHFTrainer):

    def __init__(self, args, template):
        super().__init__(args, template)
        assert args.padding_free, 'Currently `rlhf_type="rm"` only supports padding_free.'

    def loss_func(self, output_tensor, *, data):
        packed_seq_params = data.get('packed_seq_params')
        margin = data.pop('margin', None)
        num_samples = packed_seq_params.num_samples
        last_token = packed_seq_params.cu_seqlens_q[1:num_samples * 2 + 1] - 1
        rewards = output_tensor[0, last_token]
        rewards_chosen, rewards_rejected = torch.split(rewards, num_samples, dim=0)
        if margin is not None:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - margin).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected)**2)
        rewards_chosen, rewards_rejected = rewards_chosen.detach(), rewards_rejected.detach()
        metric = {
            'loss': loss.detach().clone(),
            'rewards/chosen': rewards_chosen.mean(),
            'rewards/rejected': rewards_rejected.mean(),
            'rewards/accuracies': (rewards_chosen > rewards_rejected).float().mean(),
            'rewards/margins': (rewards_chosen - rewards_rejected).mean(),
        }
        metric = self._all_reduce_metric(metric)
        return loss, metric

    def forward_step(self, data_iterator, model):
        timers = get_timers()

        # Get the batch.
        vp_stage = model.module.module.vp_stage
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()
        data.pop('loss_scale', None)
        with self.stimer:
            output_tensor = model(**data)
        return output_tensor, partial(self.loss_func, data=data)

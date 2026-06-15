# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from functools import partial
from torch import nn

from swift.utils import get_logger
from .rlhf_mixin import MegatronRLHFTrainer

logger = get_logger()


class MegatronRewardTrainer(MegatronRLHFTrainer):

    def loss_func(self, output_tensor, *, data):
        packed_seq_params = data.get('packed_seq_params')
        margin = data.pop('margin', None)
        num_samples = output_tensor.shape[0] if packed_seq_params is None else packed_seq_params.seq_lens.shape[0]
        rewards = self.get_last_tokens(output_tensor, packed_seq_params, data.get('attention_mask'))
        rewards_chosen, rewards_rejected = torch.split(rewards, num_samples // 2, dim=0)
        if margin is not None:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - margin).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if self.args.center_rewards_coefficient is not None:
            center_rewards_loss = self.args.center_rewards_coefficient * torch.mean(
                (rewards_chosen + rewards_rejected)**2)
            loss += center_rewards_loss
        rewards_chosen, rewards_rejected = rewards_chosen.detach(), rewards_rejected.detach()
        metric = {
            'loss': loss.detach().clone(),
            'rewards/chosen': rewards_chosen.mean(),
            'rewards/rejected': rewards_rejected.mean(),
            'rewards/accuracies': (rewards_chosen > rewards_rejected).float().mean(),
            'rewards/margins': (rewards_chosen - rewards_rejected).mean(),
        }
        if self.args.center_rewards_coefficient is not None:
            metric['center_rewards_loss'] = center_rewards_loss.detach()
        metric = self._all_reduce_metric(metric)
        return loss, metric

    def forward_step(self, data_iterator, model):
        vp_stage = model.module.module.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        data.pop('loss_scale', None)
        output_tensor = model(**data)
        return output_tensor, partial(self.loss_func, data=data)

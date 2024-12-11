# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Tuple

import torch

from .ppo_trainer import PPOTrainer


class RLFTTrainer(PPOTrainer):

    def __init__(self,
                 *args,
                 reward_func=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_func = reward_func

    def patch_reward(self):
        from trl.trainer import utils
        from trl.trainer import ppov2_trainer
        self.get_reward_temp = utils.get_reward
        utils.get_reward = self.get_reward
        ppov2_trainer.get_reward = self.get_reward

    def unpatch_reward(self):
        from trl.trainer import utils
        from trl.trainer import ppov2_trainer
        utils.get_reward = self.get_reward_temp
        ppov2_trainer.get_reward = self.get_reward_temp

    def get_reward(
            self, model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reward_func:
            return self.reward_func(model, query_responses, pad_token_id, context_length)
        else:
            return self.get_reward_temp(model, query_responses, pad_token_id, context_length)

    def train(self, *args, **kwargs):
        self.patch_reward()
        super().train(*args, **kwargs)
        self.unpatch_reward()

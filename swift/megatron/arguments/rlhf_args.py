# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Literal, Optional

from .sft_args import MegatronSftArguments


@dataclass
class MegatronRLHFArguments(MegatronSftArguments):
    rlhf_type: Literal['dpo', 'kto', 'grpo', 'gkd', 'rm'] = 'dpo'
    loss_scale: str = 'last_round'
    truncation_strategy: Optional[Literal['delete', 'left', 'right', 'split', None]] = None

    calculate_per_token_loss: Optional[bool] = None

    def __post_init__(self):
        if self.calculate_per_token_loss is None:
            # Ray uses a separate loss interface; retain its existing default.
            self.calculate_per_token_loss = self.rlhf_type == 'gkd' and not self.use_ray
        if (self.rlhf_type == 'gkd' and not self.use_ray and self.calculate_per_token_loss
                and self.context_parallel_size > 1):
            raise NotImplementedError('Token-normalized Megatron GKD currently requires context_parallel_size=1. '
                                      'Set calculate_per_token_loss=False to retain legacy microbatch averaging.')
        if self.rlhf_type == 'rm':
            self.task_type = 'seq_cls'
            self.num_labels = 1
        self._init_truncation_strategy()
        super().__post_init__()

    def _init_truncation_strategy(self):
        if self.truncation_strategy is not None:
            return
        if self.rlhf_type == 'grpo':
            self.truncation_strategy = 'left'
        else:
            self.truncation_strategy = 'delete'

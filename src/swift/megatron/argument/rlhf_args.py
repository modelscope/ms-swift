# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal, Optional

from .train_args import MegatronTrainArguments


@dataclass
class MegatronRLHFArguments(MegatronTrainArguments):
    rlhf_type: Literal['dpo', 'kto', 'grpo', 'rm'] = 'dpo'
    loss_scale: str = 'last_round'
    truncation_strategy: Optional[Literal['delete', 'left', 'right', 'split', None]] = None

    calculate_per_token_loss: bool = False

    def __post_init__(self):
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

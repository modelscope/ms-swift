# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal

from .train_args import MegatronTrainArguments


@dataclass
class MegatronRLHFArguments(MegatronTrainArguments):
    rlhf_type: Literal['dpo', 'kto', 'rm'] = 'dpo'
    loss_scale: str = 'last_round'

    calculate_per_token_loss: bool = False

    def __post_init__(self):
        if self.rlhf_type == 'rm':
            self.task_type = 'seq_cls'
            self.num_labels = 1
        super().__post_init__()

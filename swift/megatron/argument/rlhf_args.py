# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal

from .train_args import MegatronTrainArguments


@dataclass
class MegatronRLHFArguments(MegatronTrainArguments):
    rlhf_type: Literal['dpo', 'kto'] = 'dpo'
    loss_scale: str = 'last_round'

    calculate_per_token_loss: bool = False

    desirable_weight: float = 1.
    undesirable_weight: float = 1.
    calculate_KL: bool = True

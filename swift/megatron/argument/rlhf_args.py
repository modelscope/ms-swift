# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal

from .train_args import MegatronTrainArguments


@dataclass
class MegatronRLHFArguments(MegatronTrainArguments):
    rlhf_type: Literal['dpo'] = 'dpo'
    loss_scale: str = 'last_round'

    calculate_per_token_loss: bool = False

# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Literal

from .train_args import MegatronTrainArguments


class MegatronRLHFArguments(MegatronTrainArguments):
    rlhf_type: Literal['dpo'] = 'dpo'
    loss_scale: str = 'last_round'

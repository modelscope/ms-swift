# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from functools import partial
from typing import Literal

import torch
from megatron.core import mpu
from megatron.training import get_args, get_timers
from trl import KTOTrainer

from swift.utils import get_current_device, get_logger
from .rlhf_mixin import MegatronRLHFTrainer

logger = get_logger()


class MegatronRewardTrainer(MegatronRLHFTrainer):

    def __init__(self, args, template):
        super().__init__(args, template)
        assert args.padding_free, 'Currently `rlhf_type="rm"` only supports padding_free.'

    def loss_func(self, output_tensor):
        pass

    def forward_step(self, data_iterator, model):
        pass

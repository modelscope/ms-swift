# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel
from trl import GKDTrainer as HFGKDTrainer

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFGKDTrainer.__init__


class GKDTrainer(RLHFTrainerMixin, SwiftMixin, HFGKDTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        super().__init__(model, *_args, **kwargs)

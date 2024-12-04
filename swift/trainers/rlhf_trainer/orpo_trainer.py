# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel
from trl import ORPOTrainer as HFORPOTrainer

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFORPOTrainer.__init__
del HFORPOTrainer.get_batch_samples


class ORPOTrainer(RLHFTrainerMixin, SwiftMixin, HFORPOTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'ORPO does not require a ref_model.'
        super().__init__(model, *_args, **kwargs)

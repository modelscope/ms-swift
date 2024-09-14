# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from trl import ORPOTrainer as HFORPOTrainer

from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin

del HFORPOTrainer.__init__


class ORPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFORPOTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'ORPO does not require a ref_model.'
        super().__init__(model, *_args, **kwargs)

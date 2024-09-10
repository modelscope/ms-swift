# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel
from trl import ORPOTrainer as HFORPOTrainer

from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin
from swift.utils import get_logger

logger = get_logger()

del HFORPOTrainer.__init__


class ORPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFORPOTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'CPO does not require a ref_model.'
        super().__init__(model, *_args, **kwargs)

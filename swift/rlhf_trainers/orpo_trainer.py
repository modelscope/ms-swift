# Copyright (c) ModelScope Contributors. All rights reserved.
import torch.nn as nn
import trl
from packaging import version
from transformers import PreTrainedModel
from typing import Optional, Union

from swift.trainers import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

if version.parse(trl.__version__) >= version.parse('0.26.0'):
    from trl.experimental.orpo import ORPOTrainer as HFORPOTrainer
else:
    from trl import ORPOTrainer as HFORPOTrainer

del HFORPOTrainer.__init__


class ORPOTrainer(RLHFTrainerMixin, SwiftMixin, HFORPOTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'ORPO does not require a ref_model.'
        super().__init__(model, *_args, **kwargs)

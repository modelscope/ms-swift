# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel
from trl import CPOTrainer as HFCPOTrainer

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFCPOTrainer.__init__


class CPOTrainer(RLHFTrainerMixin, SwiftMixin, HFCPOTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'CPO/SimPO does not require a ref_model.'

        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.cpo_alpha = args.cpo_alpha
        if args.loss_type == 'simpo':
            self.simpo_gamma = args.simpo_gamma
            if self.cpo_alpha > 0:
                warnings.warn('You are using CPO-SimPO method because you set a non-zero cpo_alpha. '
                              'This will result in the CPO-SimPO method '
                              '(https://github.com/fe1ixxu/CPO_SIMPO/tree/main). '
                              'If you want to use a pure SimPO method, please set cpo_alpha to 0.')
        super().__init__(model, *_args, **kwargs)

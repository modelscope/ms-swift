# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel
from trl import CPOTrainer as HFCPOTrainer

from swift.utils import get_logger
from .mixin import RLHFTrainerMixin, SwiftMixin
from .push_to_ms import PushToMsHubMixin

logger = get_logger()

del HFCPOTrainer.__init__


class CPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFCPOTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'CPO does not require a ref_model.'

        args = kwargs['args']
        self.cpo_alpha = args.cpo_alpha
        if args.loss_type == 'simpo':
            self.simpo_gamma = args.simpo_gamma
            if self.cpo_alpha > 0:
                warnings.warn('You are using CPO-SimPO method because you set a non-zero cpo_alpha. '
                              'This will result in the CPO-SimPO method '
                              '(https://github.com/fe1ixxu/CPO_SIMPO/tree/main). '
                              'If you want to use a pure SimPO method, please set cpo_alpha to 0.')
        super().__init__(model, *_args, **kwargs)

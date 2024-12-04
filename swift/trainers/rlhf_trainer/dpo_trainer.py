# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union

import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel
from trl import DPOTrainer as HFDPOTrainer

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFDPOTrainer.__init__
del HFDPOTrainer.get_batch_samples


class DPOTrainer(RLHFTrainerMixin, SwiftMixin, HFDPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import FDivergenceConstants
        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free
        super().__init__(model, ref_model, *_args, **kwargs)

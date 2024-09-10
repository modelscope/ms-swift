# Copyright (c) Alibaba, Inc. and its affiliates.
from trl import DPOTrainer as HFDPOTrainer
from peft import PeftModel

from swift.utils import get_logger
from .mixin import RLHFTrainerMixin, SwiftMixin
from .push_to_ms import PushToMsHubMixin
from typing import  Optional, Union
import torch.nn as nn
from trl.trainer import FDivergenceConstants
from transformers import PreTrainedModel

logger = get_logger()

del HFDPOTrainer.__init__


class DPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFDPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        args = kwargs['args']
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)
        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free
        self.is_vision_model = False
        super().__init__(model, ref_model, *_args, **kwargs)

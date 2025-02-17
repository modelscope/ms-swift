# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel
from trl import KTOTrainer as HFKTOTrainer

from swift.utils import get_logger
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

logger = get_logger()

del HFKTOTrainer.__init__


class KTOTrainer(RLHFTrainerMixin, SwiftMixin, HFKTOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        args = kwargs['args']
        args.disable_dropout = True
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.is_peft_model = isinstance(model, PeftModel)
        if hasattr(args, 'loss_type'):
            self.loss_type = args.loss_type
        else:
            self.loss_type = 'kto'

        self.ref_adapter_name = None
        # Not all losses require a KL calculation
        self.calculate_KL = True
        if self.loss_type in ['apo_zero_unpaired']:
            self.calculate_KL = False
        super().__init__(model, ref_model, *_args, **kwargs)

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        is_kl = True

        def _add_data_hook(model, args, kwargs):
            nonlocal is_kl
            if is_kl:
                kwargs = {k[len('KL_completion_'):]: v for k, v in batch.items() if k.startswith('KL_completion_')}
            else:
                kwargs = {k[len('completion_'):]: v for k, v in batch.items() if k.startswith('completion_')}
            is_kl = not is_kl
            return (), kwargs

        @contextmanager
        def _patch_model_call():
            handle = model.register_forward_pre_hook(_add_data_hook, with_kwargs=True, prepend=True)

            try:
                yield
            finally:
                handle.remove()

        with _patch_model_call():
            return super().forward(model, batch)

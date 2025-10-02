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

        self.ref_adapter_name = getattr(args, 'ref_adapter_name', None)
        self.model_adapter_name = None
        # Not all losses require a KL calculation
        self.calculate_KL = True
        if self.loss_type in ['apo_zero_unpaired']:
            self.calculate_KL = False
        super().__init__(model, ref_model, *_args, **kwargs)

    # Code borrowed from huggingface/trl
    def forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL_logps = self._compute_kl_logps(model, batch)
        model_kwargs, labels = self._get_model_kwargs(batch, 'completion_')
        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True

        outputs = model(**model_kwargs)
        completion_logits = outputs.logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            labels,
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if completion_logps.shape[0] != len(batch['label']):
            raise ValueError('There is a mismatch between the number of examples in this batch and the number of '
                             'examples for which an output sequence was predicted.')

        chosen_idx = [i for i in range(completion_logps.shape[0]) if batch['label'][i] is True]
        rejected_idx = [i for i in range(completion_logps.shape[0]) if batch['label'][i] is False]

        chosen_logps = completion_logps[chosen_idx, ...]
        rejected_logps = completion_logps[rejected_idx, ...]

        chosen_logits = completion_logits[chosen_idx, ...]
        rejected_logits = completion_logits[rejected_idx, ...]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps, outputs.aux_loss)
        else:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps)

    def _get_model_kwargs(self, inputs, prefix: str):
        model_kwargs = {k[len(prefix):]: v for k, v in inputs.items() if k.startswith(prefix)}
        labels = model_kwargs['labels']
        if not self.is_encoder_decoder:
            model_kwargs.pop('labels')
        return model_kwargs, labels

    # Code borrowed from huggingface/trl (compat trl<0.17)
    def _compute_kl_logps(self, model, batch):
        """Compute KL log probabilities for a given batch."""
        KL_logps = None
        if self.calculate_KL:
            KL_model_kwargs, labels = self._get_model_kwargs(batch, 'KL_completion_')
            with torch.no_grad():
                KL_logits = model(**KL_model_kwargs).logits
            KL_logps = self.get_batch_logps(
                KL_logits,
                labels,
                average_log_prob=False,
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )
        return KL_logps

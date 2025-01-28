# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel
from trl import DPOTrainer as HFDPOTrainer

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFDPOTrainer.__init__


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
        self.use_weighting = False

        super().__init__(model, ref_model, *_args, **kwargs)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch = batch.copy()
        num_examples = batch['labels'].shape[0] // 2
        labels = batch.pop('labels', None)
        if self.is_encoder_decoder:
            batch['labels'] = labels

        if self.aux_loss_enabled:
            batch['output_router_logits'] = True
        outputs = model(**batch, use_cache=False)
        batch['labels'] = labels
        if outputs.logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            outputs.logits = outputs.logits[:, -labels.shape[1]:]
        for key in ['input_ids', 'attention_mask', 'labels']:
            batch[f'concatenated_{key}'] = batch.pop(key, None)
        if self.__class__.__name__ == 'ORPOTrainer':  # Pass-through labels
            batch['concatenated_input_ids'] = batch['concatenated_labels']

        all_logits = outputs.logits

        if all_logits.shape[:2] != batch['concatenated_labels'].shape[:2]:
            # for llava, the model returns logits for the entire sequence,
            # including the image tokens (placed before the text tokens)
            seq_len = batch['concatenated_labels'].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            batch['concatenated_labels'],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        output = {}

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        if self.args.rpo_alpha is not None:
            labels = batch['concatenated_labels'].clone()
            output['nll_loss'] = cross_entropy_loss(all_logits[:num_examples], labels[:num_examples])

        if self.loss_type == 'ipo':
            all_logps = all_logps / size_completion

        output['chosen_logps'] = all_logps[:num_examples]
        output['rejected_logps'] = all_logps[num_examples:]
        output['mean_chosen_logits'] = all_logits[:num_examples].mean()
        output['mean_rejected_logits'] = all_logits[num_examples:].mean()

        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss

        return output

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(f'Logits (batch and sequence length dim) {logits.shape[:-1]}'
                             'and labels must have the same shape {labels.shape}')
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        else:
            labels = labels.clone()

        loss_mask = labels != label_pad_token_id

        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

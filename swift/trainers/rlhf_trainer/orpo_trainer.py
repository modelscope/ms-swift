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

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        model_kwargs = batch.copy()
        labels = model_kwargs.pop('labels', None)
        len_chosen = model_kwargs['input_ids'].shape[0] // 2
        if self.is_vision_model:
            # fix llava & multi-round loss computation. (more GPU memory)
            masked_labels = labels.clone()
            masked_labels[len_chosen:] = -100
            model_kwargs['labels'] = masked_labels

        if self.is_encoder_decoder:
            model_kwargs['decoder_input_ids'] = self._shift_right(labels)

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True

        outputs = model(**model_kwargs, use_cache=False)
        all_logits = outputs.logits

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        chosen_nll_loss = outputs.loss
        if not self.is_vision_model:
            chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            labels,
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)

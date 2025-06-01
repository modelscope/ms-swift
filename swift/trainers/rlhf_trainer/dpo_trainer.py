# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer.utils import selective_log_softmax

from swift.utils import get_logger
from ..mixin import DataLoaderMixin, SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFDPOTrainer.__init__
logger = get_logger()


class DPOTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HFDPOTrainer):

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

    def get_nll_loss(self, logits, labels):
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        # Enable model parallelism
        labels = labels.to(logits.device)
        return loss_fct(logits, labels)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch = batch.copy()
        labels = batch.pop('labels', None)

        base_model = self.template.get_base_model(self.model)
        use_logits_to_keep = self.args.use_logits_to_keep
        if use_logits_to_keep is None:
            # padding_free or packing
            use_logits_to_keep = 'logits_to_keep' in inspect.signature(base_model.forward).parameters
        logger.info_once(f'use_logits_to_keep: {use_logits_to_keep}')

        if use_logits_to_keep:
            batch['logits_to_keep'] = (labels.shape[-1] -
                                       (torch.ne(labels, self.label_pad_token_id).int().argmax(-1))).max().item() + 1
            assert batch['logits_to_keep'] > 0
            labels = labels[:, -batch['logits_to_keep']:]
        num_examples = labels.shape[0] // 2
        if self.is_encoder_decoder or self.args.use_liger_kernel:
            batch['labels'] = labels.clone()

        if self.aux_loss_enabled:
            batch['output_router_logits'] = True
        # liger_kernel optimizes nll_loss more effectively.
        if 'labels' in batch:
            if self.template.padding_free:
                batch['labels'][num_examples:] = self.label_pad_token_id
            else:
                pass
        outputs = model(**batch, use_cache=False)
        all_logits = outputs.logits
        if self.template.padding_free:
            labels, all_logits = self.template.unflatten_row(labels, all_logits, batch['position_ids'])

        if all_logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            all_logits = all_logits[:, -labels.shape[1]:]

        if not self.is_encoder_decoder:
            # Shift so that tokens < n predict n
            all_logits = all_logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        loss_mask = labels != self.label_pad_token_id

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            labels,
            loss_mask,
        )

        output = {}

        if self.args.rpo_alpha is not None:
            output['nll_loss'] = self.get_nll_loss(all_logits[:num_examples], labels[:num_examples])

        if self.loss_type == 'ipo':
            all_logps = all_logps / size_completion

        output['chosen_logps'] = all_logps[:num_examples]
        output['rejected_logps'] = all_logps[num_examples:]
        output['mean_chosen_logits'] = all_logits[:num_examples][loss_mask[:num_examples]].mean()
        output['mean_rejected_logits'] = all_logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss

        return output

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        loss_mask: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(f'Logits (batch and sequence length dim) {logits.shape[:-1]}'
                             'and labels must have the same shape {labels.shape}')
        labels = labels.clone()
        labels[~loss_mask] = 0
        # https://github.com/huggingface/trl/pull/2799
        # Reduce peak vram consumption with efficient selective log_softmax
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        return per_token_logps.sum(-1), loss_mask.sum(-1)

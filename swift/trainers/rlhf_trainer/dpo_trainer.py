# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import PreTrainedModel
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.utils import RunningMoments, selective_log_softmax

from swift.llm import to_device
from swift.utils import get_logger
from ..mixin import DataLoaderMixin, SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFDPOTrainer.__init__
logger = get_logger()


def new_gather_function(tensor):
    tensor_list = gather_object([tensor])
    tensor_list = [t[None] if t.ndim == 0 else t for t in tensor_list]
    return torch.concat(to_device(tensor_list, tensor.device), dim=0)


class DPOTrainer(RLHFTrainerMixin, SwiftMixin, DataLoaderMixin, HFDPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import FDivergenceConstants
        args = kwargs['args']
        self.label_smoothing = args.label_smoothing
        if 'loss_weights' in DPOConfig.__dict__:
            # trl >= 0.20
            self.loss_type = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
            self.loss_weights = args.loss_weights
        else:
            self.loss_type = args.loss_type

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        for loss_type in loss_types:
            if (loss_type in ['hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'apo_zero', 'apo_down']
                    and args.label_smoothing > 0):
                warnings.warn(
                    f'You are using the {loss_type} loss type that does not support label smoothing. The '
                    '`label_smoothing` parameter will be ignored. '
                    'Set `label_smoothing` to `0.0` to remove this warning.',
                    UserWarning,
                )
            if loss_type == 'kto_pair':
                raise ValueError('Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.')

        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = args.ref_adapter_name
        self.model_adapter_name = None
        self.reference_free = args.reference_free
        self.use_weighting = False

        super().__init__(model, ref_model, *_args, **kwargs)

        if 'bco_pair' in loss_types:
            self.running = RunningMoments(self.accelerator)
        if self.template.packing:
            self.accelerator.gather_for_metrics = new_gather_function

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_ref_model: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch = batch.copy()

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(batch)
        if self.aux_loss_enabled:
            batch['output_router_logits'] = True
        labels = batch.pop('labels', None)
        if self.is_encoder_decoder:
            batch['labels'] = labels
        position_ids = batch.pop('_position_ids', None)
        if position_ids is None:
            position_ids = batch.get('position_ids')
        outputs = model(**batch, use_cache=False)
        all_logits = outputs.logits

        if all_logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            all_logits = all_logits[:, -labels.shape[1]:]

        if not self.is_encoder_decoder and self.template.sequence_parallel_size == 1:
            # Shift so that tokens < n predict n
            labels = torch.roll(labels, shifts=-1, dims=1)
        per_token_logps, mean_all_logits, loss_mask = self.get_per_token_logps(
            all_logits, labels, label_pad_token_id=self.label_pad_token_id)
        origin_per_token_logps = per_token_logps

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        if 'ipo' in loss_types:
            size_completion = loss_mask.sum(dim=-1)
            per_token_logps = per_token_logps / size_completion

        output = {}
        if self.template.padding_free:
            cu_seqlens = self.get_cu_seqlens(position_ids, batch.get('logits_to_keep'))
            all_logps = per_token_logps.new_zeros((cu_seqlens.shape[0] - 1, ))
            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                all_logps[i] = per_token_logps[:, start:end].sum()
            num_examples = all_logps.shape[0] // 2
            num_tokens = cu_seqlens[num_examples]
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['mean_rejected_logits'] = mean_all_logits[:, num_tokens:][loss_mask[:, num_tokens:]].mean()
        else:
            all_logps = per_token_logps.sum(-1)
            num_examples = labels.shape[0] // 2
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:num_examples][loss_mask[:num_examples]].mean()
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:num_examples][loss_mask[:num_examples]].mean()
            output['mean_rejected_logits'] = mean_all_logits[num_examples:][loss_mask[num_examples:]].mean()
        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss
        return output

    @staticmethod
    def get_per_token_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id=-100,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.shape[:-1] != labels.shape:
            raise ValueError(f'Logits (batch and sequence length dim) {logits.shape[:-1]}'
                             'and labels must have the same shape {labels.shape}')
        loss_mask = labels != label_pad_token_id
        labels = labels.clone()
        labels[~loss_mask] = 0
        # https://github.com/huggingface/trl/pull/2799
        # Reduce peak vram consumption with efficient selective log_softmax
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        return per_token_logps, logits.mean(-1), loss_mask

    def training_step(self, model, inputs, *args, **kwargs):
        inputs['_position_ids'] = inputs.get('position_ids')
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)

    def prediction_step(self, model, inputs, *args, **kwargs):
        inputs['_position_ids'] = inputs.get('position_ids')
        with self.template.forward_context(self.model, inputs):
            return super().prediction_step(model, inputs, *args, **kwargs)

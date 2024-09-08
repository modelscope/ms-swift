from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from trl import DPOTrainer as HFDPOTrainer

from swift.utils import get_logger
from .mixin import RLHFTrainerMixin, SwiftMixin
from .push_to_ms import PushToMsHubMixin

logger = get_logger()

del HFDPOTrainer.__init__


class DPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFDPOTrainer):

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal['train', 'eval'] = 'train',
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""

        metrics = {}
        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if 'reference_chosen_logps' in batch and 'reference_rejected_logps' in batch:
            reference_chosen_logps = batch['reference_chosen_logps']
            reference_rejected_logps = batch['reference_rejected_logps']
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        if self.sft_beta > 0.:
            losses = (1 - self.sft_beta) * losses + self.sft_beta * policy_nll_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = 'eval_' if train_eval == 'eval' else ''
        metrics[f'{prefix}rewards/chosen'] = chosen_rewards.mean().cpu()
        metrics[f'{prefix}rewards/rejected'] = rejected_rewards.mean().cpu()
        metrics[f'{prefix}rewards/accuracies'] = reward_accuracies.mean().cpu()
        metrics[f'{prefix}rewards/margins'] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f'{prefix}logps/rejected'] = policy_rejected_logps.detach().mean().cpu()
        metrics[f'{prefix}logps/chosen'] = policy_chosen_logps.detach().mean().cpu()
        metrics[f'{prefix}logps/ref_rejected'] = reference_rejected_logps.detach(  # noqa
        ).mean().cpu()  # noqa
        metrics[f'{prefix}logps/ref_chosen'] = reference_chosen_logps.detach().mean().cpu()
        metrics[f'{prefix}logits/rejected'] = policy_rejected_logits.detach().mean().cpu()
        metrics[f'{prefix}logits/chosen'] = policy_chosen_logits.detach().mean().cpu()
        if self.sft_beta > 0:
            metrics[f'{prefix}sft_loss'] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, 'router_aux_loss_coef', 0.0) * aux_loss, metrics

        return losses.mean(), metrics

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        res = HFDPOTrainer.concatenated_inputs(batch, is_encoder_decoder, is_vision_model, label_pad_token_id,
                                               padding_value, device)
        for k, v in batch.items():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                continue
            if k.startswith('chosen_'):
                res[k.replace('chosen_', '')] = batch[k].copy()
        for k, v in batch.items():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                continue
            if k.startswith('rejected_'):
                res[k.replace('rejected_', '')].append(batch[k][0])
        return res

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        model_kwargs = concatenated_batch.copy()
        model_kwargs['input_ids'] = model_kwargs.pop('concatenated_input_ids')
        model_kwargs['attention_mask'] = model_kwargs.pop('concatenated_attention_mask')
        model_kwargs.pop('concatenated_labels', None)
        outputs = model(
            **model_kwargs,
            use_cache=False,
        )

        @contextmanager
        def _patch_concatenated_forward():
            _old_concatenated_inputs = self.concatenated_inputs
            _old_model_call = model.__class__.__call__
            self.concatenated_inputs = lambda *args, **kwargs: concatenated_batch
            model.__class__.__call__ = lambda *args, **kwargs: outputs
            yield
            self.concatenated_inputs = _old_concatenated_inputs
            model.__class__.__call__ = _old_model_call

        with _patch_concatenated_forward():
            return super().concatenated_forward(model, batch)

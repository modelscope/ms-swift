# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from accelerate.utils import gather_object
from contextlib import contextmanager, nullcontext
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.utils.versions import require_version
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer.dpo_config import DPOConfig
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from swift.trainers import DataLoaderMixin, SwiftMixin
from swift.utils import get_logger, to_device
from .rlhf_mixin import RLHFTrainerMixin

try:
    from trl.trainer.utils import RunningMoments
except ImportError:
    # trl >= 0.29
    from trl.experimental.bco.bco_trainer import RunningMoments

_ALPHA_DIVERGENCE_COEF_KEY = 'alpha_divergence_coef'
_ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0

del HFDPOTrainer.__init__
logger = get_logger()


def _get_exp_cap(value, decimal=4):
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_log_max = torch.log(vdtype_max).to(value.device)
    return torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max


def _cap_exp(value, cap=-1):
    cap = _get_exp_cap(value) if cap < 0 else cap
    return torch.exp(torch.clamp(value, max=cap))


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
        self.f_divergence_type = getattr(args, 'f_divergence_type', 'reverse_kl')
        self.f_alpha_divergence_coef = getattr(args, 'f_alpha_divergence_coef', 0.5)
        self.f_divergence_params = {_ALPHA_DIVERGENCE_COEF_KEY: self.f_alpha_divergence_coef}
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = getattr(args, 'ref_adapter_name', None)
        self.model_adapter_name = None
        self.reference_free = getattr(args, 'reference_free', None) or False
        self.use_weighting = False

        super().__init__(model, ref_model, *_args, **kwargs)

        if 'bco_pair' in loss_types:
            self.running = RunningMoments(self.accelerator)

        if self.args.ld_alpha is not None:
            require_version('trl>=0.18', '`ld_alpha` requires that "trl>=0.18".')

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
        text_position_ids = batch.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = batch.get('position_ids')
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
            cu_seqlens = self.get_cu_seqlens(text_position_ids, batch.get('logits_to_keep'))
            num_examples = (cu_seqlens.shape[0] - 1) // 2
            all_logps = per_token_logps.new_zeros((num_examples * 2, ))
            completion_lengths = (cu_seqlens[1:] - cu_seqlens[:-1])
            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper

            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                length = end - start
                public_length = public_lengths[i % num_examples]
                if self.args.ld_alpha is not None and not is_ref_model and length > public_length:
                    front_logps = per_token_logps[:, start:start + public_length].sum()
                    rear_logps = per_token_logps[:, start + public_length:end].sum()
                    all_logps[i] = front_logps + self.args.ld_alpha * rear_logps
                else:
                    all_logps[i] = per_token_logps[:, start:end].sum()
            num_tokens = cu_seqlens[num_examples]
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:, :num_tokens][loss_mask[:, :num_tokens]].mean()
            output['mean_rejected_logits'] = mean_all_logits[:, num_tokens:][loss_mask[:, num_tokens:]].mean()
        else:
            num_examples = labels.shape[0] // 2
            if not is_ref_model:
                output['nll_loss'] = -origin_per_token_logps[:num_examples][loss_mask[:num_examples]].mean()
            if self.args.ld_alpha is not None and not is_ref_model:
                completion_lengths = loss_mask.sum(dim=1)

                chosen_lengths = completion_lengths[:num_examples]
                rejected_lengths = completion_lengths[num_examples:]
                public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
                public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

                # Use loss_mask to compute position within completion
                # cumsum gives position within completion (1-indexed), subtract 1 to get 0-indexed
                completion_position_ids = (loss_mask.cumsum(dim=1) - 1) * loss_mask

                ld_mask = completion_position_ids < public_lengths.unsqueeze(1)
                # front_mask: positions within public_lengths (shared prefix)
                # rear_mask: positions beyond public_lengths (length-dependent suffix)
                front_mask = (ld_mask & loss_mask).float()
                rear_mask = (~ld_mask & loss_mask).float()
                front_logps = (per_token_logps * front_mask).sum(dim=1)
                rear_logps = (per_token_logps * rear_mask).sum(dim=1)

                all_logps = front_logps + self.args.ld_alpha * rear_logps
            else:
                all_logps = per_token_logps.sum(-1)
            output['chosen_logps'] = all_logps[:num_examples]
            output['rejected_logps'] = all_logps[num_examples:]
            output['mean_chosen_logits'] = mean_all_logits[:num_examples][loss_mask[:num_examples]].mean()
            output['mean_rejected_logits'] = mean_all_logits[num_examples:][loss_mask[num_examples:]].mean()
        if self.aux_loss_enabled:
            output['aux_loss'] = outputs.aux_loss
        return output

    # some methods are removed in trl>=0.29, override them to compatible trl<0.29 and trl>=0.29
    # consider abort to refactor these methods to follow trl>=0.29 in the future
    @contextmanager
    def null_ref_context(self):
        with (self.accelerator.unwrap_model(self.model).disable_adapter()
              if self.is_peft_model and not self.ref_adapter_name else nullcontext()):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or 'default')

    def compute_ref_log_probs(self, batch):
        compute_ref_context_manager = (
            torch.autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext())
        with torch.no_grad(), compute_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
        return ref_model_output['chosen_logps'], ref_model_output['rejected_logps']

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        loss_type: str = 'sigmoid',
        model_output: Optional[Dict[str, torch.FloatTensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        device = self.accelerator.device

        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        if self.f_divergence_type == 'alpha_divergence':
            alpha_coef = _ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and _ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[_ALPHA_DIVERGENCE_COEF_KEY])
            logits = (_cap_exp(rejected_logratios * -alpha_coef)
                      - _cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logratios = logratios.to(device)
            ref_logratios = ref_logratios.to(device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == 'js_divergence':
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        if loss_type == 'sigmoid':
            losses = (-F.logsigmoid(self.beta * logits) *
                      (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing)

        elif loss_type == 'robust':
            losses = (-F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                      + F.logsigmoid(-self.beta * logits) * self.label_smoothing) / (1 - 2 * self.label_smoothing)

        elif loss_type == 'exo_pair':
            import math
            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (F.logsigmoid(
                self.beta * logits) - math.log(1 - self.label_smoothing)) + (-self.beta * logits).sigmoid() * (
                    F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))

        elif loss_type == 'hinge':
            losses = torch.relu(1 - self.beta * logits)

        elif loss_type == 'ipo':
            losses = (logits - 1 / (2 * self.beta))**2

        elif loss_type == 'bco_pair':
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid(
                (self.beta * chosen_logratios) - delta) - F.logsigmoid(-(self.beta * rejected_logratios - delta))

        elif loss_type == 'sppo_hard':
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta)**2 + (b + 0.5 / self.beta)**2

        elif loss_type == 'nca_pair':
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (-F.logsigmoid(chosen_rewards) - 0.5 * F.logsigmoid(-chosen_rewards)
                      - 0.5 * F.logsigmoid(-rejected_rewards))

        elif loss_type == 'aot_unpaired':
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (-F.logsigmoid(self.beta * delta) *
                      (1 - self.label_smoothing) - F.logsigmoid(-self.beta * delta) * self.label_smoothing)

        elif loss_type == 'aot':
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (-F.logsigmoid(self.beta * delta) *
                      (1 - self.label_smoothing) - F.logsigmoid(-self.beta * delta) * self.label_smoothing)

        elif loss_type == 'apo_zero':
            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)
            losses = losses_chosen + losses_rejected

        elif loss_type == 'apo_down':
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected

        elif loss_type == 'discopop':
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            logits = logits * self.beta
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation

        elif loss_type == 'sft':
            sft_loss = model_output['nll_loss']
            batch_size = chosen_logps.shape[0]
            losses = sft_loss.expand(batch_size)
            chosen_rewards = torch.zeros_like(chosen_logps)
            rejected_rewards = torch.zeros_like(rejected_logps)

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_unpaired', 'discopop', 'apo_zero', "
                "'apo_down', 'sft']")

        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        if 'ref_chosen_logps' in batch and 'ref_rejected_logps' in batch:
            ref_chosen_logps = batch['ref_chosen_logps']
            ref_rejected_logps = batch['ref_rejected_logps']
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses = 0
        chosen_rewards = 0
        rejected_rewards = 0

        loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
        loss_weights = self.loss_weights if hasattr(self, 'loss_weights') and self.loss_weights else None
        for idx, loss_type in enumerate(loss_types):
            _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                model_output['chosen_logps'],
                model_output['rejected_logps'],
                ref_chosen_logps,
                ref_rejected_logps,
                loss_type,
                model_output,
            )
            weight = loss_weights[idx] if loss_weights else 1.0
            losses = losses + _losses * weight
            chosen_rewards = chosen_rewards + _chosen_rewards * weight
            rejected_rewards = rejected_rewards + _rejected_rewards * weight

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output['nll_loss']

        if self.use_weighting:
            losses = losses * model_output['policy_weights']

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output['aux_loss']

        prefix = 'eval_' if train_eval == 'eval' else ''
        metrics[f'{prefix}rewards/chosen'] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f'{prefix}rewards/rejected'] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f'{prefix}rewards/accuracies'] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f'{prefix}rewards/margins'] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item())
        metrics[f'{prefix}logps/chosen'] = (
            self.accelerator.gather_for_metrics(model_output['chosen_logps']).detach().mean().item())
        metrics[f'{prefix}logps/rejected'] = (
            self.accelerator.gather_for_metrics(model_output['rejected_logps']).detach().mean().item())
        metrics[f'{prefix}logits/chosen'] = (
            self.accelerator.gather_for_metrics(model_output['mean_chosen_logits']).detach().mean().item())
        metrics[f'{prefix}logits/rejected'] = (
            self.accelerator.gather_for_metrics(model_output['mean_rejected_logits']).detach().mean().item())
        if self.args.rpo_alpha is not None or 'sft' in loss_types:
            metrics[f'{prefix}nll_loss'] = (
                self.accelerator.gather_for_metrics(model_output['nll_loss']).detach().mean().item())
        if self.aux_loss_enabled:
            metrics[f'{prefix}aux_loss'] = (
                self.accelerator.gather_for_metrics(model_output['aux_loss']).detach().mean().item())

        return losses.mean(), metrics

    def store_metrics(self, metrics, train_eval='train'):
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs, start_time=None):
        from transformers import Trainer
        train_eval = 'train' if 'loss' in logs else 'eval'
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        mode = 'train' if self.model.training else 'eval'
        custom_metrics = self.custom_metrics[mode]
        prefix = 'eval_' if mode == 'eval' else ''
        logs.update(self.compute_custom_metrics(custom_metrics, prefix))
        return Trainer.log(self, logs, start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        compute_loss_context_manager = (
            torch.autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext())
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval='train')

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval='train')

        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps

        if return_outputs:
            return loss, metrics
        return loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only=False, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval='eval')

            self.store_metrics(metrics, train_eval='eval')

            if prediction_loss_only:
                return loss.detach(), None, None

            logits_dict = {
                'eval_logits/chosen': metrics['eval_logits/chosen'],
                'eval_logits/rejected': metrics['eval_logits/rejected'],
            }
            logits = torch.tensor(list(logits_dict.values()), device=self.accelerator.device)
            labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

            return (loss.detach(), logits, labels)

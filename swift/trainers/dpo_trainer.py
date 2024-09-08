from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from transformers import PreTrainedModel, Trainer
from transformers.utils import is_peft_available
from trl import DPOConfig
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer import FDivergenceConstants, disable_dropout_in_model
from trl.trainer.utils import DPODataCollatorWithPadding, pad_to_length

from swift.utils import get_logger
from .mixin import SwiftMixin
from .push_to_ms import PushToMsHubMixin
from .utils import sort_by_max_length

logger = get_logger()


class DPOTrainer(PushToMsHubMixin, SwiftMixin, HFDPOTrainer):

    def __init__(self,
                 model: Union['PreTrainedModel', torch.nn.Module],
                 ref_model: Optional[Union['PreTrainedModel', torch.nn.Module]],
                 args: DPOConfig,
                 sft_beta=0.,
                 test_oom_error=False,
                 **kwargs):
        self.streaming = kwargs.pop('streaming', False)
        self.is_vision_model = kwargs.pop('is_vision', False)
        self.generate_during_eval = args.generate_during_eval
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.tokenizer = kwargs['tokenizer']
        self.lazy_tokenize = kwargs.pop('lazy_tokenize', False)
        self.sft_beta = sft_beta
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.label_smoothing = args.label_smoothing
        self.aux_loss_enabled = getattr(model.config, 'output_router_logits', False)
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}

        kwargs['data_collator'] = DPODataCollatorWithPadding(
            pad_token_id=self.tokenizer.pad_token_id,
            label_pad_token_id=args.label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)
        self.label_pad_token_id = -100
        self.padding_value = 0
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False  # ?
        kwargs['super_class'] = Trainer
        SwiftMixin.__init__(self, model, args, **kwargs)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.ref_model = ref_model
        self.ref_adapter_name = None
        self.reference_free = False
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (getattr(ref_model, 'is_loaded_in_8bit', False)
                        or getattr(ref_model, 'is_loaded_in_4bit', False)):
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if not self.streaming and not self.lazy_tokenize:
            train_ds_info = self.stat_dataset(self.train_dataset, self.is_encoder_decoder)

            if self.eval_dataset is not None:
                val_ds_info = self.stat_dataset(self.eval_dataset, self.is_encoder_decoder)
                self.dataset_info = {'train_dataset': train_ds_info, 'val_dataset': val_ds_info}
            else:
                self.dataset_info = {'train_dataset': train_ds_info}
        else:
            self.dataset_info = {}
        if test_oom_error:
            self.train_dataset = sort_by_max_length(self.train_dataset, 20000, self.is_encoder_decoder)
        # performance
        self.perf: Dict[str, Any] = {
            'gen_time': 0.,
            'gen_len': 0,
            'memory': {},
            'model': self.model.get_trainable_parameters() if hasattr(self.model, 'get_trainable_parameters') else None,
        }

    def train(self, *args, **kwargs) -> torch.Tensor:
        res = super().train(*args, **kwargs)
        for i in range(torch.cuda.device_count()):
            self.perf['memory'][f'cuda:{i}'] = f'{torch.cuda.max_memory_reserved(i)/1024/1024/1024:.2f}GiB'
        return res

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

    @staticmethod
    def stat_dataset(llm_dataset, is_encoder_decoder: bool = False) -> Any:
        _token_len = []
        from datasets import Dataset as HfDataset
        from swift.utils.np_utils import stat_array
        if isinstance(llm_dataset, HfDataset):
            if is_encoder_decoder:
                prompt = llm_dataset['prompt_input_ids']
                chosen = llm_dataset['chosen_labels']
                rejected = llm_dataset['chosen_labels']
                for p, cc, rr in zip(prompt, chosen, rejected):
                    _token_len.append(max(len(cc), len(rr)) + len(p))
            else:
                chosen = llm_dataset['chosen_input_ids']
                rejected = llm_dataset['rejected_input_ids']
                for cc, rr in zip(chosen, rejected):
                    _token_len.append(max(len(cc), len(rr)))
        else:
            for d in llm_dataset:
                if is_encoder_decoder:
                    _token_len.append(
                        max(len(d['chosen_labels']), len(d['chosen_labels'])) + len(d['prompt_input_ids']))
                else:
                    _token_len.append(max(len(d['chosen_input_ids']), len(d['rejected_input_ids'])))
        _, stat_str = stat_array(_token_len)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel, trainer, PreTrainedTokenizer
from trl import KTOTrainer as HFKTOTrainer

from swift.llm.utils.template import Context, Template, History
from swift.llm.utils.utils import sort_by_max_length
from swift.utils import get_logger
from .callback import DefaultFlowCallbackNew, PrinterCallbackNew, ProgressCallbackNew
from .mixin import PushToMsHubMixin, SwiftMixin

logger = get_logger()


class KTOTrainer(PushToMsHubMixin, SwiftMixin, HFKTOTrainer):

    def __init__(self, *args, template: Template, sft_beta=0., test_oom_error=False, **kwargs):
        self.template = template
        self.sft_beta = sft_beta
        super().__init__(*args, **kwargs)
        train_ds_info = self.stat_dataset(self.train_dataset)
        val_ds_info = self.stat_dataset(self.eval_dataset)
        self.dataset_info = {'train_dataset': train_ds_info, 'val_dataset': val_ds_info}
        if test_oom_error:
            self.train_dataset = sort_by_max_length(self.train_dataset, 20000)
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

    def encode_batch(self, batch: Dict[str, List[Any]]):
        """
        batch
        - prompt
        - completion
        - label
        maybe have
            - history
            - system
        """
        """return: inputs, tokenizer_kwargs"""

        query: Optional[str] = batch.get('prompt', None)
        response: Optional[str] = batch.get('completion', None)
        history: Optional[History] = batch.get('history', [])
        system: Optional[str] = batch.get('system', None)
        template_type = getattr(self.template, 'template_type', None)
        if len(history) > 0:
            assert self.template.support_multi_round, (
                f'The template does not support multi-round chat, template_type: {template_type}')
        if system is None:
            if self.template.use_default_system:
                system = self.template.default_system
        elif system == '':
            system = None
        else:
            assert self.template.prefix_has_system is not None, (
                f'The template does not support `system`, template_type: {template_type}')
        
        inputs, tokenizer_kwargs = self._encode(
            query, response, history, system, self.truncation_strategy, auto_add_bos=self.auto_add_bos)
        if inputs.get('labels') is None:
            inputs.pop('loss_scale', None)
        return inputs, tokenizer_kwargs

    def _tokenize(self, batch: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer") -> Dict[str, List[Any]]:
        self.encode_batch(batch)
        return super()._tokenize(batch, tokenizer)

    @staticmethod
    def stat_dataset(llm_dataset) -> Any:
        _token_len = []
        from datasets import Dataset as HfDataset
        from swift.utils.np_utils import stat_array
        if isinstance(llm_dataset, HfDataset):
            chosen = llm_dataset['chosen_input_ids']
            rejected = llm_dataset['rejected_input_ids']
            for cc, rr in zip(chosen, rejected):
                _token_len.append(max(len(cc), len(rr)))
        else:
            for d in llm_dataset:
                _token_len.append(max(len(d['chosen_input_ids']), len(d['rejected_input_ids'])))
        _, stat_str = stat_array(_token_len)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal['train', 'eval'] = 'train',
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            concatenated_batch,
        ) = self.concatenated_forward(model, batch)

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
            chosen_labels = concatenated_batch['concatenated_labels'][:batch['chosen_labels'].shape[0]]
            sft_loss = -self.get_batch_logps(policy_chosen_logits, chosen_labels, average_log_prob=True)
            if losses.shape[0] == 2 * sft_loss.shape[0]:
                sft_loss = sft_loss.repeat(2, *sft_loss.shape[1:])
            losses = (1 - self.sft_beta) * losses + self.sft_beta * sft_loss

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

        return losses.mean(), metrics

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, Dict[str, torch.LongTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch['chosen_labels'].shape[0]

        model_kwargs = ({
            'labels': concatenated_batch['concatenated_labels'],
            'decoder_input_ids': concatenated_batch.pop('concatenated_decoder_input_ids', None),
        } if self.is_encoder_decoder else {})
        all_logits = model(
            concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask'],
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch['concatenated_labels'],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, concatenated_batch)


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew

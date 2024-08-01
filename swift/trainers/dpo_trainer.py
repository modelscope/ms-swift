from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from trl import DPOTrainer as HFDPOTrainer
from trl.trainer.utils import pad_to_length

from swift.llm.utils.template import Template
from swift.llm.utils.utils import sort_by_max_length
from swift.utils import get_logger
from .mixin import PushToMsHubMixin, SwiftMixin
from .utils import build_tokenized_answer, patch_trl

logger = get_logger()
patch_trl()


class DPOTrainer(PushToMsHubMixin, SwiftMixin, HFDPOTrainer):

    def __init__(self, *args, template: Template, sft_beta=0., test_oom_error=False, **kwargs):
        self.template = template
        self.sft_beta = sft_beta
        is_vision = kwargs.pop('is_vision')
        self.keys = []
        super().__init__(*args, **kwargs)
        self.train_dataset = self.train_dataset.filter(lambda x: x['prompt_input_ids'] is not None)
        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.filter(lambda x: x['prompt_input_ids'] is not None)
        train_ds_info = self.stat_dataset(self.train_dataset, self.is_encoder_decoder)

        if self.eval_dataset is not None:
            val_ds_info = self.stat_dataset(self.eval_dataset, self.is_encoder_decoder)
            self.dataset_info = {'train_dataset': train_ds_info, 'val_dataset': val_ds_info}
        else:
            self.dataset_info = {'train_dataset': train_ds_info}
        if test_oom_error:
            self.train_dataset = sort_by_max_length(self.train_dataset, 20000)
        # performance
        self.perf: Dict[str, Any] = {
            'gen_time': 0.,
            'gen_len': 0,
            'memory': {},
            'model': self.model.get_trainable_parameters() if hasattr(self.model, 'get_trainable_parameters') else None,
        }
        self.is_vision_model = is_vision
        self.model.config.model_type = self.model.config.model_type[:-1]  # remove suffix

    def train(self, *args, **kwargs) -> torch.Tensor:
        res = super().train(*args, **kwargs)
        for i in range(torch.cuda.device_count()):
            self.perf['memory'][f'cuda:{i}'] = f'{torch.cuda.max_memory_reserved(i)/1024/1024/1024:.2f}GiB'
        return res

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:

        batch = {}
        if not self.is_encoder_decoder:
            # first: encode without response
            prompt = feature.copy()
            prompt['response'] = None
            prompt_tokens = self.template.encode(prompt)[0]

            # Skip examples that do not contain 'input_ids'
            if 'input_ids' not in prompt_tokens:
                return {k: None for k in self.keys}

            # resolve conflict in data_collator when labels are None, pop it afterwards
            prompt_tokens['labels'] = prompt_tokens['input_ids']
            # Batching image-related information for paired response using template
            prompt_tokens = [prompt_tokens] * 2
            prompt_tokens = self.template.data_collator(prompt_tokens)
            prompt_tokens.pop('labels')
            for k in prompt_tokens:
                if 'image' in k or 'pixel' in k:
                    continue
                prompt_tokens[k] = prompt_tokens[k][0]
                if isinstance(prompt_tokens[k], torch.Tensor):
                    prompt_tokens[k] = prompt_tokens[k].tolist()

            if 'pixel_values' in prompt_tokens and prompt_tokens['pixel_values'].dtype == torch.bfloat16:
                # datasets do not accept bfloat16; convert to float32.
                prompt_tokens['pixel_values'] = prompt_tokens['pixel_values'].to(torch.float32)

            if 'attention_mask' not in prompt_tokens:
                prompt_tokens['attention_mask'] = [1] * len(prompt_tokens['input_ids'])

            prompt_tokens = {f'prompt_{k}': v for k, v in prompt_tokens.items()}

            # encode with response
            chosen_tokens = build_tokenized_answer(feature['response'], self.template)
            chosen_tokens.update(prompt_tokens)

            rejected_tokens = build_tokenized_answer(feature['rejected_response'], self.template)
            rejected_tokens.update(prompt_tokens)

            longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens['prompt_input_ids']) + longer_response_length > self.max_length:
                    if self.truncation_mode == 'keep_start':
                        for k in ['prompt_input_ids', 'prompt_attention_mask']:
                            answer_tokens[k] = answer_tokens[k][:self.max_prompt_length]
                    elif self.truncation_mode == 'keep_end':
                        for k in ['prompt_input_ids', 'prompt_attention_mask']:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                    else:
                        raise ValueError(f'Unknown truncation mode: {self.truncation_mode}')

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens['prompt_input_ids']) + longer_response_length > self.max_length:
                    for k in ['input_ids', 'attention_mask']:
                        answer_tokens[k] = answer_tokens[k][:self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f'prompt_{k}'] + chosen_tokens[k]
                for k in ['input_ids', 'attention_mask']
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f'prompt_{k}'] + rejected_tokens[k]
                for k in ['input_ids', 'attention_mask']
            }
            chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
            _paddings = [self.label_pad_token_id] * len(chosen_tokens['prompt_input_ids'])
            chosen_sequence_tokens['labels'][:len(chosen_tokens['prompt_input_ids'])] = _paddings
            rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
            _paddings = [self.label_pad_token_id] * len(rejected_tokens['prompt_input_ids'])
            rejected_sequence_tokens['labels'][:len(rejected_tokens['prompt_input_ids'])] = _paddings

            for k, toks in {
                    'chosen_': chosen_sequence_tokens,
                    'rejected_': rejected_sequence_tokens,
                    '': prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == 'token_type_ids':
                        continue
                    batch[f'{k}{type_key}'] = tokens

        else:
            # encoder-decoder
            prompt = feature.copy()
            prompt['response'] = None
            prompt_tokens = self.template.encode(prompt)[0]

            # resolve conflict in data_collator when labels are None, pop it afterwards
            prompt_tokens['labels'] = prompt_tokens['input_ids']

            # Batching image-related information for paired response using template
            prompt_tokens = [prompt_tokens] * 2
            prompt_tokens = self.template.data_collator(prompt_tokens)
            prompt_tokens.pop('labels')
            for k in prompt_tokens:
                if 'image' in k or 'pixel' in k:
                    continue
                prompt_tokens[k] = prompt_tokens[k][0]
                if isinstance(prompt_tokens[k], torch.Tensor):
                    prompt_tokens[k] = prompt_tokens[k].tolist()

            if 'pixel_values' in prompt_tokens and prompt_tokens['pixel_values'].dtype == torch.bfloat16:
                # datasets do not accept bfloat16; convert to float32.
                prompt_tokens['pixel_values'] = prompt_tokens['pixel_values'].to(torch.float32)
            if 'attention_mask' not in prompt_tokens:
                prompt_tokens['attention_mask'] = [1] * len(prompt_tokens['input_ids'])

            prompt_tokens = {f'prompt_{k}': v for k, v in prompt_tokens.items()}

            # encode with response
            chosen_tokens = build_tokenized_answer(feature['response'], self.template)
            rejected_tokens = build_tokenized_answer(feature['rejected_response'], self.template)

            batch['chosen_labels'] = chosen_tokens['input_ids']
            batch['rejected_labels'] = rejected_tokens['input_ids']

            if model is not None and hasattr(model, 'prepare_decoder_input_ids_from_labels'):
                batch['rejected_decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch['rejected_labels']))
                batch['chosen_decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch['chosen_labels']))

            batch.update(prompt_tokens)
        if not self.keys:
            self.keys = (list(batch.keys()))
        return batch

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

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch['chosen_labels'].shape[0]

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs['labels'] = concatenated_batch['concatenated_labels']
            model_kwargs['decoder_input_ids'] = concatenated_batch.pop('concatenated_decoder_input_ids', None)

        if self.is_vision_model:
            model_kwargs['pixel_values'] = concatenated_batch['pixel_values'].to(model.dtype)

            if 'image_flags' in concatenated_batch:
                model_kwargs['image_flags'] = concatenated_batch['image_flags']

            if 'pixel_attention_mask' in concatenated_batch:
                model_kwargs['pixel_attention_mask'] = concatenated_batch['pixel_attention_mask']

            if 'image_sizes' in concatenated_batch:
                model_kwargs['image_sizes'] = concatenated_batch['image_sizes']

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True

        outputs = model(
            input_ids=concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask'],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch['concatenated_labels'].shape[:2]:
            # for llava, the model returns logits for the entire sequence,
            # including the image tokens (placed before the text tokens)
            seq_len = concatenated_batch['concatenated_labels'].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch['concatenated_labels'],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

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

        labels = concatenated_batch['concatenated_labels'].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == 'ipo':
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """
        Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids',
            which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            is_vision_model: Whether the model is an vision LLM.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch['chosen_labels'].shape[1], batch['rejected_labels'].shape[1])
        else:
            max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])

        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                if 'labels' in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith('_input_ids'):
                    pad_value = padding_value
                elif k.endswith('_attention_mask'):
                    pad_value = 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                if 'labels' in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith('_input_ids'):
                    pad_value = padding_value
                elif k.endswith('_attention_mask'):
                    pad_value = 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch['concatenated_input_ids'] = batch['prompt_input_ids'].repeat(2, 1).to(device=device)
            concatenated_batch['concatenated_attention_mask'] = (
                batch['prompt_attention_mask'].repeat(2, 1).to(device=device))

        # patch here
        if is_vision_model:
            pixel_values = [values for values in batch['prompt_pixel_values']]
            concatenated_batch['pixel_values'] = torch.concat(pixel_values)

            if 'prompt_image_flags' in batch:
                image_flags = [torch.tensor(flags) for flags in batch['prompt_image_flags']]
                concatenated_batch['image_flags'] = torch.concat(image_flags)

            if 'prompt_pixel_attention_mask' in batch:
                pixel_attention_mask = [mask for mask in batch['pixel_attention_mask']]
                concatenated_batch['pixel_attention_mask'] = torch.concat(pixel_attention_mask)

            if 'prompt_image_sizes' in batch:
                concatenated_batch['image_sizes'] = sum([b for b in batch['prompt_image_sizes']], start=[])

        return concatenated_batch

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

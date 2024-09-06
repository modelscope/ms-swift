from collections import defaultdict
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
        self.vision_keys = kwargs.pop('vision_keys', None)
        self.max_length = args.max_length
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
        self.use_dpo_data_collator = True
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
            # Here, we restore the _data, processing image information within the forward hook of the model.
            pair_batch_size = concatenated_batch['concatenated_input_ids'].shape[0]
            batch_size = pair_batch_size // 2
            if self.vision_keys is not None:
                _data = [dict() for _ in range(pair_batch_size)]
                for k in self.vision_keys:
                    if k in ['input_ids', 'labels']:
                        order = [i for pair in [(i, i + batch_size) for i in range(batch_size)] for i in pair]
                        _data = [{
                            **d, k: concatenated_batch[f'concatenated_{k}'][order[i]]
                        } for i, d in enumerate(_data)]
                    # for vision related data, paired response share the same one
                    elif k == 'images':
                        # convert the dtype of the images that may be converted to float32 in tokenize_row
                        model_dtype = self.accelerator.unwrap_model(model).dtype
                        _data = [{
                            **d, k: concatenated_batch[k][i // 2].to(model_dtype).unsqueeze(0)
                        } for i, d in enumerate(_data)]
                    elif k == 'pixel_values':
                        # convert the dtype of the pixel values that may be converted to float32 in tokenize_row
                        model_dtype = self.accelerator.unwrap_model(model).dtype
                        _data = [{**d, k: concatenated_batch[k][i // 2].to(model_dtype)} for i, d in enumerate(_data)]
                    else:
                        _data = [{**d, k: concatenated_batch[k][i // 2]} for i, d in enumerate(_data)]
                model_kwargs['_data'] = _data

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
            # for keys appear in _data, we leave data collector in hook
            if 'vision_pixel_values' in batch:
                concatenated_batch['pixel_values'] = batch['vision_pixel_values']

            if 'vision_image_flags' in batch:
                image_flags = [torch.tensor(flags) for flags in batch['vision_image_flags']]
                concatenated_batch['image_flags'] = image_flags

            if 'vision_pixel_attention_mask' in batch:
                pixel_attention_mask = [mask for mask in batch['vision_pixel_attention_mask']]
                concatenated_batch['pixel_attention_mask'] = pixel_attention_mask

            if 'vision_image_sizes' in batch:
                concatenated_batch['image_sizes'] = batch['vision_image_sizes']

            if 'vision_images' in batch:
                # images not in _data, we manually execute data collector here
                concatenated_batch['images'] = batch['vision_images'].squeeze(1).repeat(2, 1, 1, 1).to(device=device)

            if 'vision_tgt_sizes' in batch:
                concatenated_batch['tgt_sizes'] = batch['vision_tgt_sizes']

            if 'vision_image_bound' in batch:
                concatenated_batch['image_bound'] = batch['vision_image_bound']
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

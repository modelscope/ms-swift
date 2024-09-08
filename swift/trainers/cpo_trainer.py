from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from transformers import PreTrainedModel, Trainer
from transformers.utils import is_peft_available
from trl import CPOConfig
from trl import CPOTrainer as HFCPOTrainer
from trl.trainer import disable_dropout_in_model
from trl.trainer.utils import DPODataCollatorWithPadding

from swift.utils import get_logger
from .mixin import SwiftMixin
from .push_to_ms import PushToMsHubMixin

logger = get_logger()


class CPOTrainer(PushToMsHubMixin, SwiftMixin, HFCPOTrainer):

    def __init__(self,
                 model: Union['PreTrainedModel', torch.nn.Module],
                 args: CPOConfig,
                 test_oom_error=False,
                 **kwargs):
        kwargs.pop('ref_model', None)
        self.lazy_tokenize = kwargs.pop('lazy_tokenize', False)
        self.streaming = kwargs.pop('streaming', False)
        self.is_vision_model = kwargs.pop('is_vision', False)
        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.is_encoder_decoder = model.config.is_encoder_decoder
        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.tokenizer = kwargs['tokenizer']
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.label_smoothing = args.label_smoothing
        self.cpo_alpha = args.cpo_alpha
        if args.loss_type == 'simpo':
            self.simpo_gamma = args.simpo_gamma
            if self.cpo_alpha > 0:
                logger.warning('You are using CPO-SimPO method because you set a non-zero cpo_alpha. '
                               'This will result in the CPO-SimPO method '
                               '(https://github.com/fe1ixxu/CPO_SIMPO/tree/main). '
                               'If you want to use a pure SimPO method, please set cpo_alpha to 0.')
        self.aux_loss_enabled = getattr(model.config, 'output_router_logits', False)

        kwargs['data_collator'] = DPODataCollatorWithPadding(
            pad_token_id=self.tokenizer.pad_token_id,
            label_pad_token_id=args.label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        self.use_dpo_data_collator = True
        self.label_pad_token_id = -100
        self.padding_value = 0
        if args.disable_dropout:
            disable_dropout_in_model(model)
        self._peft_has_been_casted_to_bf16 = False
        kwargs['super_class'] = Trainer
        SwiftMixin.__init__(self, model, args, **kwargs)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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

        if self.is_vision_model:
            concatenated_batch = self.concatenated_vision_inputs(
                batch, concatenated_batch, device=self.accelerator.device)

        len_chosen = batch['chosen_labels'].shape[0]

        if self.is_encoder_decoder and self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.tokenizer.pad_token_id

        model_kwargs = ({
            'decoder_input_ids': self._shift_right(concatenated_batch['concatenated_labels']),
        } if self.is_encoder_decoder else {})

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

        if self.cpo_alpha == 0:
            nll_loss = torch.tensor(0.0).to(self.accelerator.device)
        else:
            nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch['concatenated_labels'],
            average_log_prob=self.loss_type in ['ipo', 'simpo'],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    @staticmethod
    def concatenated_vision_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        concatenated_batch: Dict[str, torch.LongTensor],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
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

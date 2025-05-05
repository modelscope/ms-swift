# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled

try:
    from trl import AutoModelForCausalLMWithValueHead
except (ImportError, RuntimeError):
    AutoModelForCausalLMWithValueHead = None


class RLHFTrainerMixin:

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import disable_dropout_in_model
        from swift.llm import HfConfigFactory
        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        args = kwargs['args']
        self.beta = getattr(args, 'beta', 0.0)
        if getattr(args, 'disable_dropout', False):
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.is_encoder_decoder = kwargs['template'].is_encoder_decoder
        self.aux_loss_enabled = getattr(model.config, 'output_router_logits', False)
        self._peft_has_been_casted_to_bf16 = False
        self.generate_during_eval = getattr(args, 'generate_during_eval', False)
        if self.is_encoder_decoder:
            self.decoder_start_token_id = HfConfigFactory.get_config_attr(model.config, 'decoder_start_token_id')
            self.pad_token_id = HfConfigFactory.get_config_attr(model.config, 'pad_token_id')
        # not use
        self.is_vision_model = False
        self.label_pad_token_id = -100
        self.use_dpo_data_collator = True
        super().__init__(model, *_args, **kwargs)
        if is_deepspeed_zero3_enabled() and ref_model is not None:
            try:
                from trl.models.utils import prepare_deepspeed
            except ImportError as e:
                raise ImportError('Please install trl>=0.14 via `pip install "trl>=0.14"`') from e
            prepare_deepspeed(self.ref_model, self.accelerator)  # Does not wrap DeepSpeedEngine
        self.padding_value = self.tokenizer.pad_token_id

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        model_kwargs = batch.copy()
        labels = model_kwargs.pop('labels', None)
        if self.is_encoder_decoder:
            model_kwargs['labels'] = labels

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True
        outputs = model(**model_kwargs, use_cache=False)
        model_kwargs['labels'] = labels
        model_kwargs['chosen_labels'] = torch.zeros(model_kwargs['labels'].shape[0] // 2)  # just get shape
        if outputs.logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            outputs.logits = outputs.logits[:, -labels.shape[1]:]
        for key in ['input_ids', 'attention_mask', 'labels']:
            model_kwargs[f'concatenated_{key}'] = model_kwargs.pop(key, None)
        if self.__class__.__name__ == 'ORPOTrainer':  # Pass-through labels
            model_kwargs['concatenated_input_ids'] = model_kwargs['concatenated_labels']

        @contextmanager
        def _patch_concatenated_forward():
            _old_concatenated_inputs = self.concatenated_inputs
            _old_model_call = model.__class__.__call__
            self.concatenated_inputs = lambda *args, **kwargs: model_kwargs
            model.__class__.__call__ = lambda *args, **kwargs: outputs
            try:
                yield
            finally:
                self.concatenated_inputs = _old_concatenated_inputs
                model.__class__.__call__ = _old_model_call

        with _patch_concatenated_forward():
            return super().concatenated_forward(model, model_kwargs)

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor, *args, **kwargs):
        if self.is_encoder_decoder:
            labels = labels.clone()  # fix trl bug
        return super().get_batch_logps(logits, labels, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        res = super().compute_loss(model, inputs, return_outputs=return_outputs)
        # compat transformers>=4.46.*
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = res[0] if return_outputs else res
            loss /= self.args.gradient_accumulation_steps
            return (loss, res[1:]) if return_outputs else loss
        return res

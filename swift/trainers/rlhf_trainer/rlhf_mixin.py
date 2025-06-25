# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from trl.models.utils import prepare_deepspeed


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
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.padding_value = self.tokenizer.pad_token_id

    def get_train_dataloader(self):
        train_dataloader = super().get_train_dataloader()
        base_dataloader = train_dataloader.base_dataloader if hasattr(
            train_dataloader, 'base_dataloader') and isinstance(train_dataloader.base_dataloader,
                                                                DataLoader) else train_dataloader
        if base_dataloader.worker_init_fn is not None and not isinstance(
                base_dataloader.worker_init_fn, partial) and 'num_workers' in inspect.signature(
                    base_dataloader.worker_init_fn).parameters:
            base_dataloader.worker_init_fn = partial(
                base_dataloader.worker_init_fn,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index)
        return train_dataloader

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        res = super().compute_loss(model, inputs, return_outputs=return_outputs)
        # compat transformers>=4.46.*
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = res[0] if return_outputs else res
            loss = loss / self.args.gradient_accumulation_steps
            return (loss, res[1:]) if return_outputs else loss
        return res

    def _get_train_sampler(self, train_dataset=None):
        get_train_sampler = super()._get_train_sampler
        parameters = inspect.signature(get_train_sampler).parameters
        kwargs = {'train_dataset': train_dataset} if 'train_dataset' in parameters else {}
        return get_train_sampler(**kwargs)

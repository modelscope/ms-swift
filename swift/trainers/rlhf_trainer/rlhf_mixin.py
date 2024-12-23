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


class ModelWrapper(nn.Module):
    # compat zero3 & rlhf
    def __init__(self, model: nn.Module, ref_model: nn.Module):
        super().__init__()
        self._model = model
        self._ref_model = ref_model

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, key: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(key)
        except AttributeError:
            if '_model' in dir(self):
                return getattr(self._model, key)
            raise

    def load_state_dict(self, *args, **kwargs):
        return self._model.load_state_dict(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._model.parameters(*args, **kwargs)

    @contextmanager
    def _save_load_context(cls, trainer):
        # fix zero3 & save/load model
        deepspeed_model = trainer.deepspeed
        _new_model = deepspeed_model._model
        _old_model = deepspeed_model.__dict__['module']
        deepspeed_model.__dict__['module'] = _new_model
        deepspeed_model._modules['module'] = _new_model
        trainer.model = _new_model
        try:
            yield
        finally:
            deepspeed_model.__dict__['module'] = _old_model
            deepspeed_model._modules['module'] = _old_model
            trainer.model = deepspeed_model


class RLHFTrainerMixin:

    @staticmethod
    def get_model_config_attr(config, key):
        for k in [None, 'language_config', 'llm_config', 'text_config']:
            if k is None:
                llm_config = config
            else:
                llm_config = getattr(config, k, None)
            if llm_config:
                val = getattr(llm_config, key)
                if val is not None:
                    return val

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        from trl.trainer import disable_dropout_in_model
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
            self.decoder_start_token_id = self.get_model_config_attr(model.config, 'decoder_start_token_id')
            self.pad_token_id = self.get_model_config_attr(model.config, 'pad_token_id')
        # not use
        self.is_vision_model = False
        self.label_pad_token_id = -100
        self.use_dpo_data_collator = True
        if is_deepspeed_zero3_enabled() and ref_model is not None:
            model = ModelWrapper(model, ref_model)
        super().__init__(model, *_args, **kwargs)
        self.padding_value = self.tokenizer.pad_token_id

    def _save_checkpoint(self, model, *args, **kwargs):
        context = nullcontext()
        if hasattr(model, '_save_load_context'):
            context = model._save_load_context(self)
        with context:
            return super()._save_checkpoint(model, *args, **kwargs)

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

    def compute_loss(self, model, inputs, return_outputs=None, num_items_in_batch=None):
        res = super().compute_loss(model, inputs, return_outputs=return_outputs)
        # compat transformers>=4.46.*
        if num_items_in_batch is not None:
            loss = res[0] if return_outputs else res
            loss /= self.args.gradient_accumulation_steps
            return (loss, res[1:]) if return_outputs else loss
        return res

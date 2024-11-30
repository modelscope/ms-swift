# Copyright (c) Alibaba, Inc. and its affiliates.
import pickle
import time
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.utils import Serializer, use_torchacc
from swift.utils.torchacc_utils import ta_trim_graph
from .mixin import SwiftMixin


class Trainer(SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(SwiftMixin, HfSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        from swift.plugin import MeanMetric
        super().__init__(*args, **kwargs)
        self._custom_metrics['acc'] = MeanMetric()

    @contextmanager
    def _patch_predict_with_generate(self):
        has_hook = self._handles
        origin_data_collator = self.data_collator
        if has_hook:
            self.template.remove_post_encode_hook()
        else:
            self.data_collator = partial(self.template.pre_data_collator, model=self.model)
        try:
            yield
        finally:
            if has_hook:
                self.template.register_post_encode_hook([self.model])
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = (self.template.mode_context('pt'),
                   self._patch_predict_with_generate()) if self.args.predict_with_generate else nullcontext()
        with context:
            return super().evaluate(*args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        labels = [
            Serializer.to_tensor(InferRequest.remove_response(data['messages']), device=self.args.device)
            for data in inputs['_data']
        ]
        resp_list = self.infer_engine.infer(
            inputs['_data'], RequestConfig(max_tokens=self.args.max_new_tokens), use_tqdm=False)
        response_list = [
            Serializer.to_tensor(resp.choices[0].message.content, device=self.args.device) for resp in resp_list
        ]

        return None, response_list, labels

    def compute_loss(self, model, inputs, return_outputs=None, num_items_in_batch=None):
        loss_kwargs = {}
        labels = None
        if (self.label_smoother is not None or self.compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')

        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                if getattr(self.args, 'average_tokens_across_devices', False):
                    outputs.loss *= self.accelerator.num_processes
                outputs.loss = outputs.loss * (labels[:, 1:] != -100).sum() / num_items_in_batch

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

        if self.args.sequence_parallel_size > 1:
            from swift.trainers.xtuner import reduce_xtuner_sequence_parallel_loss
            loss = reduce_xtuner_sequence_parallel_loss(loss, labels)

        if getattr(self.args, 'average_tokens_across_devices', False):
            loss *= self.accelerator.num_processes

        if outputs.logits is not None:
            # In case of Liger
            self._compute_token_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def _compute_token_acc(self, outputs, labels) -> None:
        from swift.plugin import compute_acc

        acc_steps = self.args.acc_steps
        preds = outputs.logits.argmax(dim=2)
        if self.state.global_step % acc_steps == 0:
            if use_torchacc():
                ta_trim_graph()
                preds = preds.to('cpu')
                labels = labels.to('cpu')
            acc_list = compute_acc(
                preds, labels, acc_strategy=self.args.acc_strategy, is_encoder_decoder=self.args.is_encoder_decoder)
            self._custom_metrics['acc'].update(acc_list)

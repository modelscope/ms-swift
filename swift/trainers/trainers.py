# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.plugin import MeanMetric, get_loss_func
from swift.utils import compute_acc, use_torchacc
from swift.utils.torchacc_utils import ta_trim_graph
from .mixin import SwiftMixin


class Trainer(SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(SwiftMixin, HfSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_metrics['acc'] = MeanMetric()

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

        inputs.pop('loss_scale', None)
        has_labels = 'labels' in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, '_gen_kwargs'):
            gen_kwargs = self._gen_kwargs.copy()
            if hasattr(self.model, 'generation_config'):
                gen_kwargs.update(self.model.generation_config.to_dict())

        if gen_kwargs.get('max_length') is None and gen_kwargs.get('max_new_tokens') is None:
            gen_kwargs['max_length'] = self.model.config.max_length
        gen_kwargs['num_beams'] = (
            gen_kwargs['num_beams'] if gen_kwargs.get('num_beams') is not None else self.model.config.num_beams)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs['synced_gpus'] = (
            gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus') is not None else default_synced_gpus)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if ('labels' in inputs and 'decoder_input_ids' in inputs
                and inputs['labels'].shape == inputs['decoder_input_ids'].shape):
            inputs = {k: v for k, v in inputs.items() if k != 'decoder_input_ids'}

        gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        # fix generate warning
        if 'max_length' in gen_kwargs and 'max_new_tokens' in gen_kwargs and gen_kwargs['max_new_tokens'] is not None:
            gen_kwargs.pop('max_length')
        gen_time = time.time()
        generate_inputs = inputs.copy()
        if has_labels:
            _labels = inputs['labels'][0]
            n_mask = 0
            for i in range(len(_labels)):
                if _labels[i] != -100:
                    n_mask = i
                    break

            for k in ['input_ids', 'attention_mask']:
                generate_inputs[k] = generate_inputs[k][:, :n_mask]
            generate_inputs['labels'] = generate_inputs['labels'][:, n_mask:]

        generated_tokens = self.model.generate(**generate_inputs, **gen_kwargs)
        gen_time = time.time() - gen_time

        if hasattr(self.model, 'encoder') and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = generate_inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = generate_inputs[self.model.main_input_name]

        generated_tokens = generated_tokens[:, generation_inputs.shape[1]:]

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get('max_length') is not None and generated_tokens.shape[-1] < gen_kwargs['max_length']:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_length'])
        elif gen_kwargs.get('max_new_tokens') is not None and generated_tokens.shape[-1] < (gen_kwargs['max_new_tokens']
                                                                                            + 1):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs['max_new_tokens'] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs['labels']).mean().detach()
                else:
                    loss = (outputs['loss'] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = generate_inputs['labels']
            if gen_kwargs.get('max_length') is not None and labels.shape[-1] < gen_kwargs['max_length']:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs['max_length'])
            elif gen_kwargs.get('max_new_tokens') is not None and labels.shape[-1] < (gen_kwargs['max_new_tokens'] + 1):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs['max_new_tokens'] + 1))
        else:
            labels = None

        return loss, generated_tokens, labels

    def compute_loss(self, model, inputs, return_outputs=None, num_items_in_batch=None):
        loss_kwargs = {}
        if (self.label_smoother is not None or self.compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')
        else:
            labels = None

        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
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
        else:
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

        if self.args.sequence_parallel_size > 1:
            from swift.trainers.xtuner import reduce_xtuner_sequence_parallel_loss
            loss = reduce_xtuner_sequence_parallel_loss(loss, labels)

        if getattr(self.args, 'average_tokens_across_devices', False):
            loss *= self.accelerator.num_processes

        self._compute_token_acc(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def _compute_token_acc(self, outputs, labels) -> None:
        acc_steps = self.args.acc_steps
        if self.state.global_step % acc_steps == 0:
            if use_torchacc():
                ta_trim_graph()
                preds = preds.to('cpu')
                masks = masks.to('cpu')
                labels = labels.to('cpu')
            acc_list = compute_acc(
                outputs.logits.argmax(dim=2),
                labels,
                acc_strategy=self.args.acc_strategy,
                is_encoder_decoder=self.args.is_encoder_decoder)
            self._custom_metrics['acc'].update(acc_list)

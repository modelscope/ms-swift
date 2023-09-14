# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers import trainer
try: 
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from .callback import DefaultFlowCallbackNew, ProgressCallbackNew
from .mixin import PushToMsHubMixin, SwiftMixin


class Trainer(PushToMsHubMixin, SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # performance
        self.perf: Dict[str, Any] = {
            'gen_time':
            0.,
            'gen_len':
            0,
            'memory': {},
            'train_time':
            0.,
            'model':
            self.model.get_trainable_parameters() if hasattr(
                self.model, 'get_trainable_parameters') else None,
        }
        self._iter_perf = 0

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        train_time = time.time()
        training_output = super().training_step(*args, **kwargs)
        train_time = time.time() - train_time
        self.perf['train_time'] = self.perf['train_time'] + train_time
        self._iter_perf += 1
        if self._iter_perf > 20 and not self.perf[
                'memory'] and torch.cuda.device_count() > 0:
            for i in range(torch.cuda.device_count()):
                self.perf['memory'][
                    f'device:{i}'] = torch.cuda.memory_reserved(i)
        return training_output

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys)

        has_labels = 'labels' in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, '_gen_kwargs'):
            gen_kwargs = self._gen_kwargs.copy()
            if hasattr(self.model, 'generation_config'):
                gen_kwargs.update(self.model.generation_config.to_dict())

        if gen_kwargs.get('max_length') is None and gen_kwargs.get(
                'max_new_tokens') is None:
            gen_kwargs['max_length'] = self.model.config.max_length
        gen_kwargs['num_beams'] = (
            gen_kwargs['num_beams'] if gen_kwargs.get('num_beams') is not None
            else self.model.config.num_beams)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs['synced_gpus'] = (
            gen_kwargs['synced_gpus'] if gen_kwargs.get('synced_gpus')
            is not None else default_synced_gpus)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if ('labels' in inputs and 'decoder_input_ids' in inputs and
                inputs['labels'].shape == inputs['decoder_input_ids'].shape):
            inputs = {
                k: v
                for k, v in inputs.items() if k != 'decoder_input_ids'
            }

        gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        gen_time = time.time()
        generated_tokens = self.model.generate(**inputs, **gen_kwargs)
        gen_time = time.time() - gen_time

        if hasattr(
                self.model, 'encoder'
        ) and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]
        gen_len = len(generated_tokens[0])
        self.perf['gen_time'] = self.perf['gen_time'] + gen_time
        self.perf['gen_len'] = self.perf['gen_len'] + gen_len

        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get('max_length') is not None and generated_tokens.shape[
                -1] < gen_kwargs['max_length']:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs['max_length'])
        elif gen_kwargs.get('max_new_tokens'
                            ) is not None and generated_tokens.shape[-1] < (
                                gen_kwargs['max_new_tokens'] + 1):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs['max_new_tokens'] + 1)

        if self.args.prediction_loss_only:
            return None, None, None

        if has_labels:
            labels = inputs['labels']
            if gen_kwargs.get('max_length') is not None and labels.shape[
                    -1] < gen_kwargs['max_length']:
                labels = self._pad_tensors_to_max_len(labels,
                                                      gen_kwargs['max_length'])
            elif gen_kwargs.get(
                    'max_new_tokens') is not None and labels.shape[-1] < (
                        gen_kwargs['max_new_tokens'] + 1):
                labels = self._pad_tensors_to_max_len(
                    labels, (gen_kwargs['max_new_tokens'] + 1))
        else:
            labels = None

        return None, generated_tokens, labels


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]

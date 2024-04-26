# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from peft import PeftModel
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers import trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import \
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_utils import seed_worker
from transformers.utils import is_peft_available, is_torch_xla_available

from swift.torchacc_utils import (ta_eval_dataloader, ta_test_dataloader,
                                  ta_train_dataloader)
from swift.utils import use_torchacc
from .callback import (DefaultFlowCallbackNew, PrinterCallbackNew,
                       ProgressCallbackNew)
from .mixin import PushToMsHubMixin, SwiftMixin

try:
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


class Trainer(PushToMsHubMixin, SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):

    def __init__(self, sequence_parallel_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # performance
        self.perf: Dict[str, Any] = {
            'gen_time':
            0.,
            'gen_len':
            0,
            'memory': {},
            'model':
            self.model.get_trainable_parameters() if hasattr(
                self.model, 'get_trainable_parameters') else None,
        }
        self._acc = torch.tensor(0.).to(self.args.device)
        if SUPPORT_XTUNER:
            from xtuner.parallel.sequence import (init_sequence_parallel,
                                                  SequenceParallelSampler,
                                                  reduce_sequence_parallel_loss,
                                                  get_sequence_parallel_world_size,
                                                  get_sequence_parallel_group)
            self.sequence_parallel_size = sequence_parallel_size
            init_sequence_parallel(sequence_parallel_size)

    def train(self, *args, **kwargs) -> torch.Tensor:
        res = super().train(*args, **kwargs)
        for i in range(torch.cuda.device_count()):
            self.perf['memory'][
                f'cuda:{i}'] = f'{torch.cuda.max_memory_reserved(i)/1024/1024/1024:.2f}GiB'
        return res

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
        # fix generate warning
        if ('max_length' in gen_kwargs and 'max_new_tokens' in gen_kwargs
                and gen_kwargs['max_new_tokens'] is not None):
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

        if hasattr(
                self.model, 'encoder'
        ) and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = generate_inputs[
                self.model.encoder.main_input_name]
        else:
            generation_inputs = generate_inputs[self.model.main_input_name]

        generated_tokens = generated_tokens[:, generation_inputs.shape[1]:]
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

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs['labels']).mean().detach()
                else:
                    loss = (outputs['loss'] if isinstance(outputs, dict) else
                            outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = generate_inputs['labels']
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

        return loss, generated_tokens, labels

    def compute_scaled_loss(self, labels: torch.Tensor,
                            lm_logits: torch.Tensor,
                            loss_scale: torch.Tensor) -> torch.Tensor:
        device = lm_logits.device
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_scale = loss_scale[..., 1:]
        # Save memory
        masks = shift_labels != -100
        shift_logits = shift_logits[masks]
        shift_labels = shift_labels[masks].to(device)
        shift_scale = shift_scale[masks].to(device)
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
        loss = shift_scale * loss
        return loss.mean()

    def compute_loss(self, model, inputs, return_outputs=None):
        assert 'labels' in inputs
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}

        labels = None
        loss_scale = None
        if 'loss_scale' in inputs:
            labels = inputs.pop('labels')
            loss_scale = inputs.pop('loss_scale')

        if self.label_smoother is not None and 'labels' in inputs:
            labels = inputs.pop('labels')

        outputs = model(**inputs)
        if loss_scale is not None:
            outputs['loss'] = self.compute_scaled_loss(labels, outputs.logits,
                                                       loss_scale)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None and loss_scale is None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        if labels is None:
            labels = inputs['labels']

        if SUPPORT_XTUNER:
            # reduce loss for logging correctly
            num_tokens = (labels != -100).sum()
            loss = reduce_sequence_parallel_loss(loss, num_tokens,
                                                 get_sequence_parallel_group())

        preds = outputs.logits.argmax(dim=2)[..., :-1]

        labels = labels[..., 1:]
        masks = labels != -100
        acc_strategy = getattr(self.args, 'acc_strategy', 'token')
        acc: Optional[Tensor] = None
        if preds.shape != labels.shape:
            pass
        elif acc_strategy == 'sentence':
            acc_list = []
            for i, m in enumerate(masks):
                acc_list.append(
                    torch.all(preds[i, m] == labels[i,
                                                    m]).to(torch.int64).item())
            acc = torch.tensor(acc_list, device=preds.device).float().mean()
        else:
            acc = (torch.masked_select(preds, masks) == torch.masked_select(
                labels, masks)).float().mean()
        if model.training and acc is not None:
            if 'acc' not in self._custom_metrics:
                self._custom_metrics['acc'] = self._acc
            self._custom_metrics['acc'] = self._custom_metrics[
                'acc'] + acc / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss

    # Support logging cuda memory usage
    # hacky: Override Trainer's private method
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch,
                                 ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs['loss'] = round(
                tr_loss_scalar /
                (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs['grad_norm'] = grad_norm.detach().item() if isinstance(
                    grad_norm, torch.Tensor) else grad_norm
            logs['learning_rate'] = self._get_learning_rate()
            logs['memory'] = get_max_cuda_memory()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith('eval_'):
                    metric_to_check = f'eval_{metric_to_check}'
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def get_train_dataloader(self):

        if not use_torchacc():
            # modified from HFTrainer.get_train_dataloader
            # RandomSampler -> SequenceParallelSampler
            if trainer.is_datasets_available():
                import datasets
            if self.train_dataset is None:
                raise ValueError('Trainer: training requires a train_dataset.')

            train_dataset = self.train_dataset
            data_collator = self.data_collator
            if trainer.is_datasets_available() and isinstance(
                    train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(
                    train_dataset, description='training')
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description='training')

            dataloader_params = {
                'batch_size': self._train_batch_size,
                'collate_fn': data_collator,
                'num_workers': self.args.dataloader_num_workers,
                'pin_memory': self.args.dataloader_pin_memory,
                'persistent_workers': self.args.dataloader_persistent_workers,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params['sampler'] = SequenceParallelSampler(
                    train_dataset, seed=1024)
                dataloader_params['drop_last'] = self.args.dataloader_drop_last
                dataloader_params['worker_init_fn'] = seed_worker

            return DataLoader(train_dataset, **dataloader_params)

        else:
            if trainer.is_datasets_available():
                import datasets

            if self.train_dataset is None:
                raise ValueError('Trainer: training requires a train_dataset.')

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(
                    train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(
                    train_dataset, description='training')
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description='training')

            return ta_train_dataloader(train_dataset, data_collator,
                                       self._get_train_sampler(), self.args,
                                       self._train_batch_size)

    def get_eval_dataloader(self, eval_dataset):
        if not use_torchacc():
            return super().get_eval_dataloader(eval_dataset)
        else:
            import torchacc as ta
            if trainer.is_datasets_available():
                import datasets

            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError(
                    'Trainer: evaluation requires an eval_dataset.')
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(
                    eval_dataset, datasets.Dataset):
                eval_dataset = self._remove_unused_columns(
                    eval_dataset, description='evaluation')
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description='evaluation')

            return ta_eval_dataloader(eval_dataset, data_collator,
                                      self._get_eval_sampler(eval_dataset),
                                      self.args)

    def get_test_dataloader(self, test_dataset):
        if not use_torchacc():
            return super().get_test_dataloader(test_dataset)
        else:
            import torchacc as ta
            if trainer.is_datasets_available():
                import datasets

            data_collator = self.data_collator

            if trainer.is_datasets_available() and isinstance(
                    test_dataset, datasets.Dataset):
                test_dataset = self._remove_unused_columns(
                    test_dataset, description='test')
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description='test')

            return ta_test_dataloader(test_dataset, data_collator,
                                      self._get_eval_sampler(test_dataset),
                                      self.args)


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew

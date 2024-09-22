# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
from typing import Dict

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from transformers.trainer_callback import (DefaultFlowCallback, ProgressCallback, TrainerCallback, TrainerControl,
                                           TrainerState)
from transformers.trainer_utils import IntervalStrategy, has_length, speed_metrics
from trl import AutoModelForCausalLMWithValueHead

from swift.utils import append_to_jsonl, get_logger, is_pai_training_job, use_torchacc
from ..utils.utils import format_time
from .arguments import TrainingArguments

logger = get_logger()


class ProgressCallbackNew(ProgressCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(desc='Train', total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0
        self.start_time = time.time()
        if use_torchacc():
            self.warmup_start_time = 0
            self.warmup_metric = None
            self.metric_warmup_step = int(args.metric_warmup_step
                                          * state.max_steps) if args.metric_warmup_step < 1 else args.metric_warmup_step

    def on_prediction_step(self, args, state: TrainerState, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                if self.training_bar is not None:
                    self.training_bar.fp.write('\n')
                self.prediction_bar = tqdm(
                    desc='Val', total=len(eval_dataloader), leave=True, dynamic_ncols=True, position=0)
            self.prediction_bar.update()

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        logs['global_step/max_steps'] = f'{state.global_step}/{state.max_steps}'
        train_percentage = state.global_step / state.max_steps if state.max_steps else 0.
        logs['percentage'] = f'{train_percentage * 100:.2f}%'
        elapsed = time.time() - self.start_time
        elapsed = max(0., elapsed)
        logs['elapsed_time'] = format_time(elapsed)
        logs['remaining_time'] = format_time(elapsed / train_percentage - elapsed)

        if use_torchacc():
            if state.global_step >= self.metric_warmup_step and self.warmup_start_time == 0:
                self.warmup_start_time = time.time()
                self.metric_warmup_step = state.global_step
            if state.max_steps == state.global_step and self.warmup_metric is None:
                num_steps = state.max_steps - self.metric_warmup_step
                num_total_samples = args.train_dataset_sample
                num_after_warmup_samples = int(num_total_samples / state.max_steps * num_steps)
                self.warmup_metric = speed_metrics('warmup_train', self.warmup_start_time, num_after_warmup_samples,
                                                   num_steps)
                self.warmup_metric['num_total_samples'] = num_total_samples
                self.warmup_metric['num_after_warmup_samples'] = num_after_warmup_samples
            if 'train_samples_per_second' in logs:
                logs.update(self.warmup_metric)
                state.log_history[-1] = logs
        for k, v in logs.items():
            if isinstance(v, float):
                logs[k] = round(logs[k], 8)
        if not is_pai_training_job() and state.is_world_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, logs)
        super().on_log(args, state, control, logs, **kwargs)
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.refresh()


class DefaultFlowCallbackNew(DefaultFlowCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        # save the last ckpt
        evaluation_strategy = args.eval_strategy if hasattr(args, 'eval_strategy') else args.evaluation_strategy
        if state.global_step == state.max_steps:
            if evaluation_strategy != IntervalStrategy.NO:
                control.should_evaluate = True
            if args.save_strategy != IntervalStrategy.NO:
                control.should_save = True
        return control


class PrinterCallbackNew(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs['global_step'] = state.global_step
        for k, v in logs.items():
            if isinstance(v, float):
                logs[k] = round(logs[k], 8)
        if not is_pai_training_job() and state.is_world_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, logs)

        _ = logs.pop('total_flos', None)
        if state.is_world_process_zero:
            print(logs, flush=True)


# This code is adapted from the LlamaFactory.
def fix_valuehead_checkpoint(model: 'AutoModelForCausalLMWithValueHead', output_dir: str,
                             safe_serialization: bool) -> None:
    r"""
    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, 'model.safetensors')
        from safetensors import safe_open
        from safetensors.torch import save_file
        with safe_open(path_to_checkpoint, framework='pt', device='cpu') as f:
            state_dict: Dict[str, torch.Tensor] = {key: f.get_tensor(key) for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, 'pytorch_model.bin')
        state_dict: Dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location='cpu')

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith('v_head.'):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace('pretrained_model.', '', 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization)

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, 'value_head.safetensors'), metadata={'format': 'pt'})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, 'value_head.bin'))

    logger.info('Value head model saved at: {}'.format(output_dir))


class FixValueHeadModelCallback(TrainerCallback):
    r"""
    A callback for fixing the checkpoint for valuehead models.
    """

    def on_save(self, args: 'TrainingArguments', state: 'TrainerState', control: 'TrainerControl', **kwargs):
        r"""
        Event called after a checkpoint save.
        """
        if args.should_save:
            output_dir = os.path.join(args.output_dir, '{}-{}'.format('checkpoint', state.global_step))
            fix_valuehead_checkpoint(
                model=kwargs.pop('model'), output_dir=output_dir, safe_serialization=args.save_safetensors)

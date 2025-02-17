# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time

from tqdm import tqdm
from transformers import trainer
from transformers.trainer_callback import (DefaultFlowCallback, PrinterCallback, ProgressCallback, TrainerControl,
                                           TrainerState)
from transformers.trainer_utils import IntervalStrategy, has_length, speed_metrics

from swift.utils import append_to_jsonl, is_pai_training_job, use_torchacc
from ..utils.utils import format_time
from .arguments import TrainingArguments


def add_train_message(logs, state, start_time) -> None:
    logs['global_step/max_steps'] = f'{state.global_step}/{state.max_steps}'
    train_percentage = state.global_step / state.max_steps if state.max_steps else 0.
    logs['percentage'] = f'{train_percentage * 100:.2f}%'
    elapsed = time.time() - start_time
    logs['elapsed_time'] = format_time(elapsed)
    if train_percentage != 0:
        logs['remaining_time'] = format_time(elapsed / train_percentage - elapsed)
    for k, v in logs.items():
        if isinstance(v, float):
            logs[k] = round(logs[k], 8)


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

        add_train_message(logs, state, self.start_time)
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


class PrinterCallbackNew(PrinterCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return super().on_train_begin(args, state, control, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        add_train_message(logs, state, self.start_time)
        if not is_pai_training_job() and state.is_world_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, logs)

        _ = logs.pop('total_flos', None)
        if state.is_world_process_zero:
            print(logs, flush=True)


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew

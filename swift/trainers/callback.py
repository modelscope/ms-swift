# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import time

import torch
from tqdm import tqdm
from transformers import trainer
from transformers.trainer_callback import (DefaultFlowCallback, PrinterCallback, ProgressCallback, TrainerControl,
                                           TrainerState)
from transformers.trainer_utils import IntervalStrategy, has_length
from transformers.utils import is_torch_npu_available

from swift.utils import append_to_jsonl, format_time, get_device_count, get_logger, is_mp, is_pai_training_job
from .arguments import TrainingArguments

logger = get_logger()


def get_max_cuda_memory() -> float:
    devices = list(range(get_device_count())) if is_mp() else [None]
    mems = [torch.cuda.max_memory_reserved(device=device) for device in devices]
    return sum(mems) / 1024**3


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
    state.max_memory = max(getattr(state, 'max_memory', 0), get_max_cuda_memory())
    if not is_torch_npu_available():
        logs['memory(GiB)'] = round(state.max_memory, 2)

    logs['train_speed(iter/s)'] = round(state.global_step / elapsed, 6)


class ProgressCallbackNew(ProgressCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(desc='Train', total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0
        self.start_time = time.time()

    def on_prediction_step(self, args, state: TrainerState, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                if self.training_bar is not None:
                    self.training_bar.fp.write('\n')
                self.prediction_bar = tqdm(
                    desc='Val', total=len(eval_dataloader), leave=True, dynamic_ncols=True, position=0)
            self.prediction_bar.update()

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
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

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control = super().on_epoch_end(args, state, control, **kwargs)
        evaluation_strategy = args.eval_strategy if hasattr(args, 'eval_strategy') else args.evaluation_strategy
        if args.max_epochs is not None and args.max_epochs <= math.ceil(state.epoch):
            logger.info('Training has reached `max_epochs`. The model will be saved and the training will be exited.')
            if evaluation_strategy != IntervalStrategy.NO:
                control.should_evaluate = True
            if args.save_strategy != IntervalStrategy.NO:
                control.should_save = True
            control.should_training_stop = True
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

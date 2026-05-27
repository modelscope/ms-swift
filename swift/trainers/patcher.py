# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
import time
from tqdm import tqdm
from transformers import trainer
from transformers.trainer_callback import (DefaultFlowCallback, PrinterCallback, ProgressCallback, TrainerControl,
                                           TrainerState)
from transformers.trainer_utils import IntervalStrategy, has_length
from swift.utils import get_env_args, get_device_count, append_to_jsonl, format_time, get_logger, get_max_reserved_memory, is_pai_training_job
from .arguments import TrainingArguments
logger = get_logger()


def add_train_message(logs, state, start_time, start_step) -> None:
    logs['global_step/max_steps'] = f'{state.global_step}/{state.max_steps}'
    elapsed = time.time() - start_time
    logs['elapsed_time'] = format_time(elapsed)
    n_steps = state.global_step - start_step
    train_speed = elapsed / n_steps if n_steps > 0 else 0.0
    logs['remaining_time'] = format_time((state.max_steps - state.global_step) * train_speed)
    for k, v in logs.items():
        if isinstance(v, float):
            logs[k] = round(logs[k], 8)
    state.max_memory = max(getattr(state, 'max_memory', 0), get_max_reserved_memory())
    if state.max_memory:
        logs['memory(GiB)'] = round(state.max_memory, 2)
    logs['train_speed(s/it)'] = round(train_speed, 6)


class ProgressCallbackNewWithMFU(ProgressCallback):
    def __init__(self):
        super().__init__()
        self.device_tflops = None
        self.elapsed = 0.0
        self.step_start_time = None
        
    def on_init_end(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):

        # Top priority. Specify by ENV
        tflops = get_env_args('DEVICE_TFLOPS', float, None)
        device_count = max(get_device_count(), 1)
        self.device_tflops = tflops * device_count
        super().on_init_end(args, state, control, **kwargs)
    def on_step_begin(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.time()
        super().on_step_begin(args, state, control, **kwargs)
        
    def on_step_end(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        self.elapsed += time.time() - self.step_start_time
        super().on_step_end(args, state, control, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(desc='Train', total=state.max_steps, dynamic_ncols=True)
        self.start_step = state.global_step
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
        add_train_message(logs, state, self.start_time, self.start_step)
        total_flos = getattr(state, 'total_flos', 0)
        if self.elapsed > 0 and self.device_tflops:
            actual_flops = total_flos / self.elapsed
            theoretical_max_flops = self.device_tflops * 1e12
            mfu = actual_flops / theoretical_max_flops
        else:
            mfu = 0.0
        logger.debug(f'Total_flos[{total_flos}] elapsed_time[{self.elapsed}]sec Average MFU[{mfu}]')
        logs['MFU'] = round(mfu, 6)
        if not is_pai_training_job() and state.is_world_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, logs)
        super().on_log(args, state, control, logs, **kwargs)
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.refresh()

class ProgressCallbackNew(ProgressCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(desc='Train', total=state.max_steps, dynamic_ncols=True)
        self.start_step = state.global_step
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
        add_train_message(logs, state, self.start_time, self.start_step)
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
        self.start_step = state.global_step
        return super().on_train_begin(args, state, control, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        add_train_message(logs, state, self.start_time, self.start_step)
        if not is_pai_training_job() and state.is_world_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, logs)

        _ = logs.pop('total_flos', None)
        if state.is_world_process_zero:
            print(logs, flush=True)
# monkey patching
tflops = get_env_args('DEVICE_TFLOPS', float, None)
if tflops is not None:
    trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNewWithMFU
else:
    trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew

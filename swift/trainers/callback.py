# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import time
from collections import defaultdict
from typing import Dict, Optional

import torch.distributed as dist
from tqdm import tqdm
from transformers import trainer
from transformers.trainer_callback import (DefaultFlowCallback, PrinterCallback, ProgressCallback, TrainerCallback,
                                           TrainerControl, TrainerState)
from transformers.trainer_utils import IntervalStrategy, has_length

from swift.utils import append_to_jsonl, format_time, get_device_count, get_logger, is_mp, is_pai_training_job
from swift.utils.torch_utils import get_torch_device
from .arguments import TrainingArguments

logger = get_logger()


def get_max_reserved_memory() -> float:
    devices = list(range(get_device_count())) if is_mp() else [None]
    try:
        mems = [get_torch_device().max_memory_reserved(device=device) for device in devices]
    except AttributeError:
        return 0  # fix mps
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
    state.max_memory = max(getattr(state, 'max_memory', 0), get_max_reserved_memory())
    if state.max_memory:
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


class DatasetProgressCallback(TrainerCallback):
    """Callback for tracking per-dataset training progress in multi-dataset training.

    This callback tracks how many samples from each dataset have been consumed during training,
    and reports the progress percentage to TensorBoard.

    The callback reads progress counts from the template's `dataset_progress_counts` attribute,
    which is updated during data collation.

    Args:
        template: The template instance that tracks dataset progress counts.
        dataset_sizes: A dict mapping dataset source names to their total sample counts.
                      If None, only sample counts (not percentages) will be reported.
    """

    def __init__(self, template, dataset_sizes: Optional[Dict[str, int]] = None):
        self.template = template
        self.dataset_sizes = dataset_sizes or {}

    def on_train_begin(self, args, state, control, **kwargs):
        # Enable progress tracking in template
        self.template.track_dataset_progress = True
        self.template.dataset_progress_counts.clear()

    def _gather_counts(self) -> Dict[str, int]:
        """Gather counts from all processes in distributed training."""
        local_counts = dict(self.template.dataset_progress_counts)

        if not dist.is_initialized():
            return local_counts

        world_size = dist.get_world_size()
        if world_size == 1:
            return local_counts

        # Gather all local counts to rank 0
        gathered = [None] * world_size
        dist.gather_object(local_counts, gathered if dist.get_rank() == 0 else None, dst=0)

        if dist.get_rank() != 0:
            return {}

        # Aggregate counts from all processes
        global_counts: Dict[str, int] = defaultdict(int)
        for local in gathered:
            if local:
                for source, count in local.items():
                    global_counts[source] += count

        return dict(global_counts)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Only report on main process
        if not state.is_world_process_zero:
            return

        global_counts = self._gather_counts()

        # Calculate and log progress for each dataset
        for source, count in global_counts.items():
            total = self.dataset_sizes.get(source)
            if total and total > 0:
                progress = min(count / total * 100, 100.0)
                logs[f'dataset_progress/{source}'] = round(progress, 2)
            else:
                # If total is unknown, just log the count
                logs[f'dataset_samples/{source}'] = count


# monkey patching
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
trainer.PrinterCallback = PrinterCallbackNew

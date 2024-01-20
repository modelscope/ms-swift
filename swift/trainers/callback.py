# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import json
from tqdm.auto import tqdm
from transformers.trainer_callback import (DefaultFlowCallback,
                                           ProgressCallback, TrainerCallback,
                                           TrainerControl, TrainerState)
from transformers.trainer_utils import IntervalStrategy, has_length

from swift.trainers import TrainingArguments


class ProgressCallbackNew(ProgressCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(
                desc='Train', total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_prediction_step(self,
                           args,
                           state: TrainerState,
                           control,
                           eval_dataloader=None,
                           **kwargs):
        if state.is_local_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                if self.training_bar is not None:
                    self.training_bar.fp.write('\n')
                self.prediction_bar = tqdm(
                    desc='Val',
                    total=len(eval_dataloader),
                    leave=True,
                    dynamic_ncols=True,
                    position=0)
            self.prediction_bar.update()

    def on_log(self,
               args: TrainingArguments,
               state: TrainerState,
               control,
               logs=None,
               **kwargs):
        logs['global_step'] = state.global_step
        for k, v in logs.items():
            if isinstance(v, float):
                logs[k] = round(logs[k], 8)
        if state.is_local_process_zero and self.training_bar is not None:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(logs) + '\n')
        super().on_log(args, state, control, logs, **kwargs)
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.refresh()


class DefaultFlowCallbackNew(DefaultFlowCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        # save the last ckpt
        if state.global_step == state.max_steps:
            if args.evaluation_strategy != IntervalStrategy.NO:
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
        if state.is_local_process_zero:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(logs) + '\n')

        _ = logs.pop('total_flos', None)
        if state.is_local_process_zero:
            print(logs, flush=True)

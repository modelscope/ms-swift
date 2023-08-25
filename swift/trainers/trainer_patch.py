import os

import json
from tqdm.auto import tqdm
from transformers.trainer_callback import (DefaultFlowCallback,
                                           ProgressCallback, TrainerControl,
                                           TrainerState)
from transformers.trainer_utils import has_length

from swift.trainers import TrainingArguments


class ProgressCallbackNew(ProgressCallback):

    def on_prediction_step(self,
                           args,
                           state: TrainerState,
                           control,
                           eval_dataloader=None,
                           **kwargs):
        if state.is_local_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.training_bar.fp.write('\n')
                self.prediction_bar = tqdm(
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
        if 'learning_rate' in logs:
            logs['learning_rate'] = round(logs['learning_rate'], 8)
        if state.is_local_process_zero and self.training_bar is not None:
            jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(logs) + '\n')
        super().on_log(args, state, control, logs, **kwargs)
        if state.is_local_process_zero:
            self.training_bar.refresh()


class DefaultFlowCallbackNew(DefaultFlowCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        control = super().on_step_end(args, state, control, **kwargs)
        # save the last ckpt
        if state.global_step == state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control

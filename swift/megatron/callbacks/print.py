# Copyright (c) ModelScope Contributors. All rights reserved.
import os

import torch
from tqdm import tqdm

from swift.utils import JsonlWriter, format_time, is_master
from .base import MegatronCallback


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.jsonl_writer = None

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not is_master(), desc='Train: ')
        self.start_step = state.iteration
        self.start_time = time.time()
        logging_path = os.path.join(self.args.save, 'logging.jsonl')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')

    def on_train_end(self):
        self.training_bar.close()
        self.training_bar = None

    def on_train_step(self):
        self.training_bar.update()

    def on_eval_begin(self):
        self.eval_bar = tqdm(total=self.argseval_iters, dynamic_ncols=True, disable=not is_master(), desc='Evaluate: ')

    def on_eval_end(self):
        self.eval_bar.close()
        self.eval_bar = None

    def on_eval_step(self):
        self.eval_bar.update()

    def on_log(self, logs):
        state = self.state
        args = self.args
        logs['iteration'] = f'{state.iteration}/{args.train_iters}'
        elapsed = time.time() - self.start_time
        logs['elapsed_time'] = format_time(elapsed)
        n_steps = state.iteration - self.start_step
        train_speed = elapsed / n_steps if n_steps > 0 else 0.0
        logs['remaining_time'] = format_time((args.train_iters - state.iteration) * train_speed)
        logs['memory(GiB)'] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        logs['train_speed(s/it)'] = round(train_speed, 6)
        self.jsonl_writer.append(logs)
        self.training_bar.write(str(logs))

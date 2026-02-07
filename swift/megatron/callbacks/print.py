# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.utils import is_master
from .base import MegatronCallback


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not is_master(), desc='Train: ')

    def on_train_end(self):
        self.training_bar.close()
        self.training_bar = None

    def on_train_step(self):
        self.training_bar.update()

    def on_eval_begin(self):
        self.eval_bar = tqdm(total=eval_iters, dynamic_ncols=True, disable=not is_master(), desc='Evaluate: ')

    def on_eval_end(self):
        self.eval_bar.close()
        self.eval_bar = None

    def on_eval_step(self):
        self.eval_bar.update()

    def on_log(self, logs):
        print()

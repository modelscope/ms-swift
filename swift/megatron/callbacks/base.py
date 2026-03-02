# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swift.megatron.trainers import BaseMegatronTrainer


class MegatronCallback:

    def __init__(self, trainer: 'BaseMegatronTrainer'):
        self.trainer = trainer
        self.args = trainer.args
        self.state = trainer.state

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass

    def on_log(self, logs):
        pass

    def on_eval_begin(self):
        pass

    def on_eval_end(self):
        pass

    def on_eval_step(self):
        pass

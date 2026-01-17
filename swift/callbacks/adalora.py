# Copyright (c) ModelScope Contributors. All rights reserved.
import types
from typing import TYPE_CHECKING

from .base import TrainerCallback

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer


class AdaloraCallback(TrainerCallback):

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)
        self.global_step = 0
        self.args = args

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, _args, state, control, **kwargs):
        model = kwargs['model']
        model.peft_config['default'].total_step = state.max_steps

        def zero_grad(_self, *args, **kwargs):
            _self.update_and_allocate(self.global_step + 1)
            _self._zero_grad(*args, **kwargs)

        model._zero_grad = model.zero_grad
        model.zero_grad = types.MethodType(zero_grad, model)

    def on_step_end(self, _args, state, control, **kwargs):
        self.global_step = state.global_step

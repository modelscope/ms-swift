# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

import numpy as np
from transformers import TrainerControl, TrainerState

from swift.utils import get_logger
from .base import TrainerCallback

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer

logger = get_logger()


class EarlyStopCallback(TrainerCallback):
    """An early stop implementation"""

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)
        self.best_metric = None
        self.interval = 0
        self.total_interval = args.early_stop_interval

    def on_save(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        operator = np.greater if args.greater_is_better else np.less
        if self.best_metric is None or operator(state.best_metric, self.best_metric):
            self.best_metric = state.best_metric
            self.interval = 0
        else:
            self.interval += 1

        if self.interval >= self.total_interval:
            logger.info(f'Training stop because of eval metric is stable at step {state.global_step}')
            control.should_training_stop = True

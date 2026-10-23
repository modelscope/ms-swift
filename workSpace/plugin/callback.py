# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from swift.utils import get_logger

logger = get_logger()


class EarlyStopCallback(TrainerCallback):
    """An early stop implementation"""

    def __init__(self, total_interval=3):
        self.best_metric = None
        self.interval = 0
        self.total_interval = total_interval

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        operator = np.greater if args.greater_is_better else np.less
        if self.best_metric is None or operator(state.best_metric, self.best_metric):
            self.best_metric = state.best_metric
        else:
            self.interval += 1

        if self.interval >= self.total_interval:
            logger.info(f'Training stop because of eval metric is stable at step {state.global_step}')
            control.should_training_stop = True


extra_callbacks = []
# This example shows a simple example of EarlyStop Callback, uncomment this to use
# extra_callbacks = [EarlyStopCallback()]

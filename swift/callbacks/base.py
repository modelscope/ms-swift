# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import TrainerCallback as HfTrainerCallback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swift.trainers import Trainer, TrainingArguments


class TrainerCallback(HfTrainerCallback):

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        self.args = args
        self.trainer = trainer

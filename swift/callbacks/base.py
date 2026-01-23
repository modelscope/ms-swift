# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from transformers import TrainerCallback as HfTrainerCallback

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer


class TrainerCallback(HfTrainerCallback):

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        self.args = args
        self.trainer = trainer

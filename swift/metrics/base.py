# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

import torch
from transformers.trainer_utils import EvalPrediction

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer


class EvalMetrics(ABC):

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        self.args = args
        self.trainer = trainer

    @abstractmethod
    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        pass

    def preprocess_logits_for_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return logits

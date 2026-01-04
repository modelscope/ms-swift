from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments


class BaseLoss(ABC):

    def __init__(self, args: 'TrainingArguments'):
        self.args = args

    @abstractmethod
    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs) -> torch.Tensor:
        # You need to return a scalar representing the loss.
        pass

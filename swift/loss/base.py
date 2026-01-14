from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer


class BaseLoss(ABC):
    """Abstract base class for custom loss functions.

    This class provides a common interface for implementing custom loss functions
    that can be integrated with the ms-swift training framework. All custom loss
    implementations should inherit from this class and implement the __call__ method.

    Attributes:
        args (TrainingArguments): Training configuration and hyperparameters.
        trainer (Trainer): Reference to the trainer instance for accessing model
            and training state.
    """

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        """Initialize the loss function with training arguments and trainer.

        Args:
            args (TrainingArguments): Training configuration and hyperparameters.
            trainer (Trainer): Reference to the trainer instance.
        """
        self.args = args
        self.trainer = trainer

    @abstractmethod
    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs) -> torch.Tensor:
        """Calculate the loss value.

        This method must be implemented by all subclasses to define the specific
        loss calculation logic.

        Args:
            outputs: Model outputs.
            labels: Ground truth labels or targets.
            num_items_in_batch (int, optional): Number of items (tokens) in the current batch,
                Defaults to None.
            loss_scale (float, optional): Scaling factor to apply to the loss value.
                Defaults to None.

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss value.
        """
        pass

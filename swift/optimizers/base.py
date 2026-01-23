from typing import TYPE_CHECKING

from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer


class OptimizerCallback:
    """
    Callback for creating and managing optimizer and learning rate scheduler.

    This callback provides hooks for customizing the creation of optimizers and
    learning rate schedulers during the training process. It delegates to the
    trainer's methods by default but can be subclassed to implement custom
    optimization strategies.

    Args:
        args (TrainingArguments): The training arguments containing hyperparameters
            and configuration settings.
        trainer (Trainer): The trainer instance that will use this callback.
    """

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        self.args = args
        self.trainer = trainer

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """
        Create both optimizer and learning rate scheduler for training.

        This method initializes the optimizer and scheduler by calling their
        respective creation methods and assigns them to the trainer instance.

        Args:
            num_training_steps (int): The total number of training steps, used
                for scheduler configuration (e.g., warmup steps, decay schedule).

        Returns:
            None: The optimizer and scheduler are set directly on the trainer.
        """
        trainer = self.trainer
        trainer.optimizer = self.create_optimizer()
        trainer.scheduler = self.create_scheduler(num_training_steps, trainer.optimizer)

    def create_optimizer(self) -> Optimizer:
        return self.trainer.create_optimizer()

    def create_scheduler(self, num_training_steps: int, optimizer: Optimizer) -> LRScheduler:
        return self.trainer.create_scheduler(num_training_steps, optimizer)

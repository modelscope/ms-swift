from torch.optim import Optimizer
from transformers.trainer import Trainer

from .base import OptimizerCallback


class LorapOptimizerCallback(OptimizerCallback):

    def create_optimizer(self) -> Optimizer:
        args = self.args
        model = self.trainer.model
        optimizer_grouped_parameters = None
        if hasattr(model, 'create_optimizer_param_groups'):
            # Lora+ parameter groups
            optimizer_grouped_parameters = model.create_optimizer_param_groups(
                lr=args.learning_rate, weight_decay=args.weight_decay)

        if optimizer_grouped_parameters is None:
            # Default parameter groups
            decay_parameters = Trainer.get_decay_parameter_names(None, model)
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                    'weight_decay': args.weight_decay,
                },
                {
                    'params':
                    [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                    'weight_decay': 0.0,
                },
            ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

from transformers import Trainer

from swift.llm.utils import calculate_max_steps
from swift.trainers.optimizers.galore import create_optimizer_and_scheduler


def create_galore_optimizers(model, args):
    training_steps = calculate_max_steps(args)
    return create_optimizer_and_scheduler(
        model,
        args.training_args,
        args.galore_config,
        training_steps,
        lr=args.learning_rate,
        weight_decay=args.weight_decay)


def create_lorap_optimizers(model, args):
    training_steps = calculate_max_steps(args)
    args = args.training_args
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
                'params':
                    [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                'weight_decay':
                    args.weight_decay,
            },
            {
                'params':
                    [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                'weight_decay':
                    0.0,
            },
        ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


def default_create_optimizers(model, args):
    return None, None


optimizers_map = {
    'galore': create_galore_optimizers,
    'lorap': create_lorap_optimizers,
    'default': default_create_optimizers
}

# Copyright (c) Alibaba, Inc. and its affiliates.
import math

from transformers import Trainer

from swift.trainers.optimizers.galore import create_optimizer_and_scheduler
from swift.utils import get_dist_setting


def calculate_max_steps(args: 'TrainArguments', dataset) -> int:
    if args.max_steps and args.max_steps > 0:
        max_steps = args.max_steps
    else:
        assert not args.streaming
        len_dataset = len(dataset)
        _, _, world_size, _ = get_dist_setting()
        total_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        num_update_steps_per_epoch = len_dataset // total_train_batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    return max_steps


def create_galore_optimizers(args, model, dataset):
    training_steps = calculate_max_steps(args, dataset)
    return create_optimizer_and_scheduler(
        model,
        args.training_args,
        args.galore_config,
        training_steps,
        lr=args.learning_rate,
        weight_decay=args.weight_decay)


def create_lorap_optimizers(args, model, dataset):
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
                'params': [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                'weight_decay': args.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                'weight_decay': 0.0,
            },
        ]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


def default_create_optimizers(args, model, dataset):
    return None, None


# Add your own optimizers here, use --optimizer xxx to train
optimizers_map = {
    'galore': create_galore_optimizers,
    'lorap': create_lorap_optimizers,
    'default': default_create_optimizers
}

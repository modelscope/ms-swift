# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import sys

from transformers import Trainer

from swift.trainers.optimizers.galore import create_optimizer_and_scheduler
from swift.utils import get_dist_setting


def calculate_max_steps(args: 'TrainArguments', dataset) -> int:
    if args.max_steps and args.max_steps > 0:
        max_steps = args.max_steps
    else:
        len_dataset = len(dataset)
        _, _, world_size, _ = get_dist_setting()
        total_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        num_update_steps_per_epoch = len_dataset // total_train_batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    return max_steps


def create_galore_optimizer(args, model, dataset):
    training_steps = calculate_max_steps(args, dataset)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model, args, args.galore_config, training_steps, lr=args.learning_rate, weight_decay=args.weight_decay)
    # trainer cannot serialize galore_config
    args.galore_config = None
    return optimizer, lr_scheduler


def create_lorap_optimizer(args, model, dataset):
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


def create_muon_optimizer(args, model, dataset):
    from swift.llm import git_clone_github, get_model_arch
    if not args.local_repo_path:
        args.local_repo_path = git_clone_github('https://github.com/MoonshotAI/Moonlight.git')
    sys.path.append(os.path.join(args.local_repo_path, 'examples'))
    from toy_train import Muon

    # parse args.optim_args
    optim_args = {}
    if args.optim_args:
        for mapping in args.optim_args.replace(' ', '').split(','):
            key, value = mapping.split('=')
            optim_args[key] = value

    model_arch = get_model_arch(model.model_meta.model_arch)
    embed_key = model_arch.embedding or 'embed_tokens'
    lm_head_key = model_arch.lm_head or 'lm_head'
    muon_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2 and embed_key not in n and lm_head_key not in n
    ]
    adamw_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not (p.ndim >= 2 and embed_key not in n and lm_head_key not in n)
    ]

    return Muon(
        lr=args.learning_rate,
        wd=args.weight_decay,
        muon_params=muon_params,
        adamw_params=adamw_params,
        adamw_betas=(args.adam_beta1, args.adam_beta2),
        adamw_eps=args.adam_epsilon,
        **optim_args,
    ), None


# Add your own optimizers here, use --optimizer xxx to train
optimizers_map = {
    'galore': create_galore_optimizer,
    'lorap': create_lorap_optimizer,
    'muon': create_muon_optimizer,
}

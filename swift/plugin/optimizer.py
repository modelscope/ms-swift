# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch.nn as nn
from peft import PeftModel
from transformers import Trainer

from swift.trainers.optimizers.galore import create_optimizer_and_scheduler
from swift.utils import get_dist_setting

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments


def calculate_max_steps(args: 'TrainingArguments', dataset) -> int:
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


def create_galore_optimizer(args: 'TrainingArguments', model, dataset):
    training_steps = calculate_max_steps(args, dataset)
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model, args, args.galore_config, training_steps, lr=args.learning_rate, weight_decay=args.weight_decay)
    # trainer cannot serialize galore_config
    args.galore_config = None
    return optimizer, lr_scheduler


def create_lorap_optimizer(args: 'TrainingArguments', model, dataset):
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


def create_muon_optimizer(args: 'TrainingArguments', model, dataset):
    from swift.llm import git_clone_github
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

    model_arch = model.model_meta.model_arch
    embed_key = getattr(model_arch, 'embedding', None) or 'embed_tokens'
    lm_head_key = getattr(model_arch, 'lm_head', None) or 'lm_head'
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


def get_param_startswith(model,
                         chosen_prefix: List[str],
                         rejected_prefix: Optional[List[str]] = None) -> List[Tuple[str, nn.Parameter]]:
    chosen_prefix = chosen_prefix or []
    rejected_prefix = rejected_prefix or []
    res = []
    if not chosen_prefix:
        return res
    is_peft_model = isinstance(model, PeftModel)
    if is_peft_model:
        model = model.model
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_rejected = False
        for prefix in rejected_prefix:
            if n.startswith(prefix):
                is_rejected = True
                break
        if is_rejected:
            continue
        for prefix in chosen_prefix:
            if n.startswith(prefix):
                if is_peft_model:
                    n = f'base_model.model.{n}'
                res.append((n, p))
                break
    return res


def create_multimodal_optimizer(args: 'TrainingArguments', model, dataset):
    """ViT/Aligner/LLM use different learning rates."""
    decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
    model_arch = model.model_meta.model_arch
    vit_parameters = get_param_startswith(model, model_arch.vision_tower, model_arch.aligner)
    aligner_parameters = get_param_startswith(model, model_arch.aligner)
    llm_parameters = get_param_startswith(model, model_arch.language_model)
    optimizer_grouped_parameters = []
    for lr, parameters in zip([args.vit_lr, args.aligner_lr, args.learning_rate],
                              [vit_parameters, aligner_parameters, llm_parameters]):
        if lr is None:
            lr = args.learning_rate
        for use_wd, wd in zip([False, True], [0., args.weight_decay]):
            if use_wd:
                params = [p for n, p in parameters if n in decay_parameters]
            else:
                params = [p for n, p in parameters if n not in decay_parameters]
            if not params:
                continue
            optimizer_grouped_parameters.append({
                'params': params,
                'weight_decay': wd,
                'lr': lr,
            })
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs), None


# Add your own optimizers here, use --optimizer xxx to train
optimizers_map = {
    'galore': create_galore_optimizer,
    'lorap': create_lorap_optimizer,
    'muon': create_muon_optimizer,
    'multimodal': create_multimodal_optimizer,
}

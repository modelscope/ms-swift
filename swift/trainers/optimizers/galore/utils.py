# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Any, Tuple, List, Union, Dict
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import TrainingArguments, get_scheduler

from transformers import Trainer
from swift.tuners.module_mapping import MODEL_KEYS_MAPPING
from swift.utils import get_logger

logger = get_logger()


@dataclass
class GaLoreConfig:
    """
    The configuration class for the Galore module.


    See https://arxiv.org/abs/2403.03507

    Args:
        model_type (`str`): The model_type of Galore
        rank (`int`): The galore rank
        target_modules (`Union[str, List[str]]`): The target modules to use, if `None`,
            will use all attn and mlp linears
        update_proj_gap(`int`): The projection update interval for galore
        proj_type(`str`) The project type of Galore, valid values are `std`,
            `reverse_std`, `right`, `left`, `full`
        galore_scale(float): the scale of gradient
        optim_per_parameter(bool): Gives one optimizer per parameter
    """
    model_type: str = None
    rank: int = 128
    target_modules: Union[str, List[str]] = None
    update_proj_gap: int = 50
    galore_scale: float = 1.0
    proj_type: str = 'std'
    with_embedding: bool = False
    optim_per_parameter: bool = False


class GaloreOptimizerWrapper(Optimizer):

    def __init__(self, optimizers: Dict[Any, Optimizer]):
        self.optimizers = optimizers
        super().__init__([torch.tensor([1., 2., 3.])], {"lr": 1.})

    def zero_grad(self, *args, **kwargs) -> None:
        for optim in self.optimizers.values():
            optim.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs) -> None:
        for optim in self.optimizers.values():
            optim.step(*args, **kwargs)


class GaloreSchedulerWrapper(LRScheduler):

    def __init__(self, lr_schedulers: Dict[Any, LRScheduler]):
        self.lr_schedulers = lr_schedulers

    def step(self, *args, **kwargs) -> None:
        for lr_scheduler in self.lr_schedulers.values():
            lr_scheduler.step(*args, **kwargs)
        self._last_lr = lr_scheduler.get_last_lr()


def create_optimizer_and_scheduler(model: nn.Module, args: TrainingArguments,
                                   config: GaLoreConfig, max_steps, **defaults):
    if not config.target_modules:
        if config.model_type in MODEL_KEYS_MAPPING:
            target_modules_list = [
                MODEL_KEYS_MAPPING[config.model_type].attention.split('.{}.')[1],
                MODEL_KEYS_MAPPING[config.model_type].mlp.split('.{}.')[1]
            ]
            config.target_modules = target_modules_list
            if config.with_embedding:
                embedding = MODEL_KEYS_MAPPING[config.model_type].embedding
                idx = embedding.rfind('.')
                embedding = embedding[idx + 1:]
                target_modules_list.append(embedding)


    galore_params = []
    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)) or \
                not any(target_key in module_name for target_key in config.target_modules):
            continue

        if not module.weight.requires_grad:
            continue

        logger.info(f'Enable GaLore for weights in module: {module_name}')
        galore_params.append(module.weight)
    id_galore_params = [id(p) for p in galore_params]
    galore_defaults = {'rank': config.rank, 'update_proj_gap': config.update_proj_gap,
                       'scale': config.galore_scale, 'proj_type': config.proj_type, **defaults}
    optim_cls, optim_kwargs = get_optimizer(args)

    if config.optim_per_parameter:
        optimizer_dict = {}
        galore_defaults['update_proj_gap'] = galore_defaults['update_proj_gap'] * 2
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = optim_cls([{'params': [p], **galore_defaults}], **optim_kwargs)
                else:
                    optimizer_dict[p] = optim_cls([{'params': [p], **defaults}], **optim_kwargs)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = get_scheduler(
                    optimizer=optimizer_dict[p],
                    name=args.lr_scheduler_type,
                    num_training_steps=max_steps * 2,
                    num_warmup_steps=args.warmup_steps * 2,
                    scheduler_specific_kwargs=args.lr_scheduler_kwargs,
                )

        return GaloreOptimizerWrapper(optimizer_dict), GaloreSchedulerWrapper(scheduler_dict)
    else:
        decay_parameters = Trainer.get_decay_parameter_names(Trainer, model)
        param_groups = [{
            'params': galore_params,
            **galore_defaults,
        }]
        param_groups.extend([
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if (n in decay_parameters and id(p) not in
                        id_galore_params and p.requires_grad)
                ],
                'weight_decay': defaults['weight_decay'],
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if (n not in decay_parameters and id(p) not in
                        id_galore_params and p.requires_grad)
                ],
                'weight_decay': 0.0,
            },
        ])
        optim = optim_cls(param_groups, **optim_kwargs)
        scheduler = get_scheduler(
            optimizer=optim,
            name=args.lr_scheduler_type,
            num_training_steps=max_steps,
            num_warmup_steps=args.warmup_steps,
            scheduler_specific_kwargs=args.lr_scheduler_kwargs,
        )
        return optim, scheduler


def get_optimizer(
        args: TrainingArguments) -> Tuple[Any, Any]:
    # parse args.optim_args
    optim_args = {}
    if args.optim_args:
        for mapping in args.optim_args.replace(' ', '').split(','):
            key, value = mapping.split('=')
            optim_args[key] = value

    optimizer_kwargs = {'lr': args.learning_rate}

    adam_kwargs = {
        'betas': (args.adam_beta1, args.adam_beta2),
        'eps': args.adam_epsilon,
    }
    if args.optim == 'adafactor':
        from .adafactor import GaLoreAdafactor
        optimizer_cls = GaLoreAdafactor
        optimizer_kwargs.update({
            'scale_parameter': False,
            'relative_step': False
        })
    elif args.optim in ('adamw_hf', 'adamw_torch'):
        from .adamw import GaLoreAdamW
        optimizer_cls = GaLoreAdamW
        optimizer_kwargs.update(adam_kwargs)
    elif 'adamw' in args.optim and '8bit' in args.optim:
        try:
            from .adamw8bit import GaLoreAdamW8bit
            optimizer_cls = GaLoreAdamW8bit
            optimizer_kwargs.update(adam_kwargs)
            optimizer_kwargs.update({
                'optim_bits': 8,
                'is_paged': 'paged' in args.optim
            })
        except ImportError:
            raise ValueError(
                'Trainer tried to instantiate bnb optimizer but bnb is not installed!'
            )
    else:
        raise ValueError(
            f'Galore not supported for optimizer type: {args.optim}')
    return optimizer_cls, optimizer_kwargs

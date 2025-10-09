# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from transformers import Trainer, TrainingArguments, get_scheduler

from swift.utils import get_logger

try:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import LRScheduler

logger = get_logger()


@dataclass
class GaLoreConfig:
    """
    The configuration class for the Galore module.


    See https://arxiv.org/abs/2403.03507

    Args:
        rank (`int`): The galore rank
        target_modules (`Union[str, List[str]]`): The target modules to use, if `None`,
            will use all attn and mlp linears
        update_proj_gap(`int`): The projection update interval for galore
        proj_type(`str`) The project type of Galore, valid values are `std`,
            `reverse_std`, `right`, `left`, `full`
        galore_scale(float): the scale of gradient
        optim_per_parameter(bool): Gives one optimizer per parameter
    """
    rank: int = 128
    target_modules: Union[str, List[str]] = None
    update_proj_gap: int = 50
    galore_scale: float = 1.0
    proj_type: str = 'std'
    optim_per_parameter: bool = False
    quantize: bool = False
    proj_quant: bool = False
    proj_bits: int = 4
    proj_group_size: int = 256
    cos_threshold: float = 0.4
    gamma_proj: int = 2
    queue_size: int = 5


class GaloreOptimizerWrapper(Optimizer):

    def __init__(self, optimizers: Dict[Any, Optimizer]):
        self.optimizers = optimizers
        super().__init__([torch.tensor([1., 2., 3.])], {'lr': 1.})

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


def create_optimizer_and_scheduler(model: nn.Module, args: TrainingArguments, config: GaLoreConfig, max_steps,
                                   **defaults):
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
    galore_defaults = {
        'rank': config.rank,
        'update_proj_gap': config.update_proj_gap,
        'scale': config.galore_scale,
        'proj_type': config.proj_type,
        **defaults
    }
    if config.quantize:
        galore_defaults['quant'] = config.proj_quant
        galore_defaults['quant_n_bit'] = config.proj_bits
        galore_defaults['quant_group_size'] = config.proj_group_size
        galore_defaults['cos_threshold'] = config.cos_threshold
        galore_defaults['gamma_proj'] = config.gamma_proj
        galore_defaults['queue_size'] = config.queue_size
    optim_cls, optim_kwargs = get_optimizer(args, config)

    if config.optim_per_parameter and not config.quantize:
        # q-galore does not support optim_per_parameter
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
                    if (n in decay_parameters and id(p) not in id_galore_params and p.requires_grad)
                ],
                'weight_decay':
                defaults['weight_decay'],
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if (n not in decay_parameters and id(p) not in id_galore_params and p.requires_grad)
                ],
                'weight_decay':
                0.0,
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


def get_optimizer(args: TrainingArguments, config: GaLoreConfig) -> Tuple[Any, Any]:
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
        optimizer_kwargs.update({'scale_parameter': False, 'relative_step': False})
    elif args.optim in ('adamw_hf', 'adamw_torch', 'adamw_torch_fused'):
        if config.quantize:
            assert importlib.util.find_spec('q_galore_torch') is not None, \
                'Please install q-galore by `pip install q_galore_torch`'
            logger.info('If you encounter `absmax2` error, please downgrade your bitsandbytes to 0.40.0')
            from swift.utils import get_dist_setting
            _, _, world_size, _ = get_dist_setting()
            if world_size > 1:
                # from q_galore_torch import QGaLoreAdamW8bit_simulate as GaLoreAdamW
                from q_galore_torch import QGaLoreAdamW8bit as GaLoreAdamW
            else:
                from q_galore_torch import QGaLoreAdamW8bit as GaLoreAdamW
        else:
            from .adamw import GaLoreAdamW
        optimizer_cls = GaLoreAdamW
        optimizer_kwargs.update(adam_kwargs)
    elif 'adamw' in args.optim and '8bit' in args.optim:
        try:
            from .adamw8bit import GaLoreAdamW8bit
            optimizer_cls = GaLoreAdamW8bit
            optimizer_kwargs.update(adam_kwargs)
            optimizer_kwargs.update({'optim_bits': 8, 'is_paged': 'paged' in args.optim})
        except ImportError:
            raise ValueError('Trainer tried to instantiate bnb optimizer but bnb is not installed!')
    else:
        raise ValueError(f'Galore not supported for optimizer type: {args.optim}')
    return optimizer_cls, optimizer_kwargs

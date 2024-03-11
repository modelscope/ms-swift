# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
from dataclasses import dataclass
from typing import Tuple, Any

import torch
from packaging import version
from torch import nn
from transformers import TrainingArguments, is_bitsandbytes_available

from swift.tuners.module_mapping import MODEL_KEYS_MAPPING
from swift.tuners.utils import SwiftAdapter, SwiftConfig, SwiftOutput
from swift.utils.logger import get_logger

logger = get_logger()


@dataclass
class GaloreConfig(SwiftConfig):
    """
    The configuration class for the Galore module.


    See https://arxiv.org/abs/2403.03507

    Args:
        model_type (`str`): The model_type of Galore
        rank (`int`): The galore rank
        update_proj_gap(`int`): The projection update interval for galore
        proj_type(`str`) The project type of Galore, valid values are `std`,
            `reverse_std`, `right`, `left`, `full`
        galore_scale(float): the scale of gradient
    """
    model_type: str = None
    rank: int = 128
    update_proj_gap: int = 50
    galore_scale: float = 1.0
    proj_type: str = 'std'
    embedding: bool = False

    def __post_init__(self):
        from swift.tuners.mapping import SwiftTuners
        self.swift_type = SwiftTuners.NEFTUNE


class Galore(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: GaloreConfig,
                      adapter_name: str) -> SwiftOutput:
        """Prepare a model with `GaloreConfig`"""

        def state_dict_callback(state_dict, adapter_name):
            return state_dict

        def mark_trainable_callback(model):
            return

        def optimizer_group_callback(model, **defaults):
            galore_params = []
            if config.model_type in MODEL_KEYS_MAPPING:
                target_modules_list = [MODEL_KEYS_MAPPING[config.model_type].attention.split('.{}.')[1],
                                       MODEL_KEYS_MAPPING[config.model_type].mlp.split('.{}.')[1]]
                if config.embedding:
                    embedding = MODEL_KEYS_MAPPING[config.model_type].embedding
                    idx = embedding.rfind('.')
                    embedding = embedding[idx+1:]
                    target_modules_list.append(embedding)
            else:
                raise ValueError(f'Cannot find model type : {config.model_type}')

            galore_params = []
            names = set()
            if config.model_type in MODEL_KEYS_MAPPING:
                target_modules_list = [MODEL_KEYS_MAPPING[config.model_type].attention.split('.{}.')[1],
                                       MODEL_KEYS_MAPPING[config.model_type].mlp.split('.{}.')[1]]
            else:
                target_modules_list = ["attn", "mlp"]
            for module_name, module in model.named_modules():
                if not isinstance(module, nn.Linear) or \
                        not any(target_key in module_name for target_key in target_modules_list):
                    continue

                logger.info('enable GaLore for weights in module: ', module_name)
                galore_params.append(module.weight)
                names.add(module_name + '.weight')
            param_groups = [{'params': galore_params, 'rank': config.rank, 'update_proj_gap': config.update_proj_gap,
                             'scale': config.galore_scale, 'proj_type': config.proj_type, **defaults}]
            return names, param_groups

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback, optimizer_group_callback)

    @staticmethod
    def get_optimizer_cls_and_kwargs_galore(args: TrainingArguments) -> Tuple[Any, Any]:
        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == 'galore_adafactor':
            from .adafactor import Adafactor as GaLoreAdafactor
            optimizer_cls = GaLoreAdafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == 'galore_adamw':
            from .adamw import AdamW as GaLoreAdamW
            optimizer_cls = GaLoreAdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == 'galore_adamw8bit':
            try:
                from .adamw8bit import AdamW8bit as GaLoreAdamW8bit
                optimizer_cls = GaLoreAdamW8bit
                optimizer_kwargs.update(adam_kwargs)
                optimizer_kwargs.update({"optim_bits": 8})
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            if is_bitsandbytes_available() and version.parse(
                    importlib.metadata.version("bitsandbytes")
            ) < version.parse("0.41.1"):
                logger.warning(
                    "You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
                    "It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
                )
        else:
            raise ValueError(f'Galore not supported for optimizer type: {args.optim}')
        return optimizer_cls, optimizer_kwargs

    @staticmethod
    def activate_adapter(module: torch.nn.Module,
                         adapter_name: str,
                         activate: bool,
                         offload: str = None):
        logger.warn('Galore does not support activate or deactivate')
        pass

    @staticmethod
    def freeze_model():
        return False

    @staticmethod
    def has_additional_modules():
        return False

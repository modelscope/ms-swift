# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field

import torch
from torch import nn

from swift.utils.logger import get_logger
from .utils import SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class NEFTuneConfig(SwiftConfig):
    """
    The configuration class for the NEFTune module.

    NEFTune adds slightly noises to embedding outputs.
    See https://arxiv.org/abs/2310.05914

    Args:
        noise_alpha(`float`): The noise alpha value used for the NEFTune, default 5.0
    """
    noise_alpha: float = field(default=5.0, metadata={'help': 'The noise alpha value used for the NEFTune'})

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.NEFTUNE


class NEFTune(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: NEFTuneConfig, adapter_name: str) -> SwiftOutput:
        """Prepare a model with `NEFTuneConfig`"""
        for sub_module in model.modules():
            if isinstance(sub_module, torch.nn.Embedding):

                def neftune_hook(module, args, output):
                    if module.training and getattr(module, 'nef_activated'):
                        dims = torch.tensor(output.size(-1) * output.size(-2))
                        mag_norm = config.noise_alpha / torch.sqrt(dims)
                        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
                    return output

                if hasattr(sub_module, 'nef_activated'):
                    raise ValueError('NEFTune does not support a second tuner.')

                sub_module.register_forward_hook(neftune_hook)
                sub_module.nef_activated = True

        def state_dict_callback(state_dict, adapter_name):
            return state_dict

        def mark_trainable_callback(model):
            return

        return SwiftOutput(
            config=config, state_dict_callback=state_dict_callback, mark_trainable_callback=mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        for sub_module in module.modules():
            if isinstance(sub_module, torch.nn.Embedding):
                sub_module.nef_activated = activate

    @staticmethod
    def freeze_model():
        return False

    @staticmethod
    def has_additional_modules():
        return False

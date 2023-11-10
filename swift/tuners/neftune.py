# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import re
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from itertools import repeat
from typing import List, Union

import torch
from torch import nn

from swift.utils.logger import get_logger
from ..utils.torch_utils import find_sub_module
from .utils import ActivationMixin, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class NEFTuneConfig(SwiftConfig):
    """
    The configuration class for the side module.

    Side-Tuning only needs to train one side network and
    weights the output of pre-trained model and side network.
    'Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks'
    by Zhang et al.(2019)
    See https://arxiv.org/abs/1912.13503

    Args:
        noise_alpha(`float`): The noise alpha value used for the NEFTune, default 5.0
    """
    noise_alpha: float = field(
        default=5.0,
        metadata={
            'help':
            'The noise alpha value used for the NEFTune'
        })

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.NEFTUNE


class NEFTune:

    @staticmethod
    def prepare_model(model: nn.Module, config: NEFTuneConfig,
                      adapter_name: str) -> SwiftOutput:
        """Prepare a model with `NEFTuneConfig`"""
        for sub_module in model.modules():
            if isinstance(sub_module, torch.nn.Embedding):
                def noised_embed(orig_embed, noise_alpha):
                    def new_func(x):
                        # during training, we add noise to the embedding
                        # during generation, we don't add noise to the embedding
                        if model.training and getattr(orig_embed, 'nef_activated'):
                            embed_init = orig_embed.forward_origin(x)
                            dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                            mag_norm = noise_alpha / torch.sqrt(dims)
                            return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
                        else:
                            return orig_embed.forward_origin(x)

                    return new_func

                if hasattr(sub_module, 'nef_activated'):
                    raise ValueError(f'NEFTune does not support a second tuner.')

                sub_module.forward_origin = sub_module.forward
                sub_module.forward = noised_embed(sub_module, config.noise_alpha)
                sub_module.nef_activated = True

        def state_dict_callback(state_dict, adapter_name):
            return state_dict

        def mark_trainable_callback(model):
            return

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str,
                         activate: bool):
        for sub_module in module.modules():
            if isinstance(sub_module, torch.nn.Embedding):
                sub_module.nef_activated = activate

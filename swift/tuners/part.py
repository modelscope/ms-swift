# Copyright (c) Alibaba, Inc. and its affiliates.

import re
import types
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch import nn

from swift import get_logger
from swift.utils.torch_utils import find_sub_module
from .utils import ActivationMixin, SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class PartConfig(SwiftConfig):
    """
    Freeze the model and train a part of it.

    Args:
        target_modules(`Optional[str]`): The target modules to be trained in regex format
    """

    target_modules: Optional[str] = None

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.PART


class Part(SwiftAdapter):

    @staticmethod
    def target_module_matched(module_key: str, config: PartConfig):
        return re.fullmatch(config.target_modules, module_key)

    @staticmethod
    def prepare_model(model: nn.Module, config: PartConfig, adapter_name: str):

        def state_dict_callback(state_dict, adapter_name):
            return {key: value for key, value in state_dict.items() if Part.target_module_matched(key, config)}

        def mark_trainable_callback(model: nn.Module):
            for name, module in model.named_modules():
                module: nn.Module
                if Part.target_module_matched(name, config):
                    module.requires_grad_(True)

        return SwiftOutput(config, state_dict_callback, mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        pass

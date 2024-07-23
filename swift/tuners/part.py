# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from modelscope.hub.utils.utils import get_cache_dir
from torch import nn

from swift import get_logger
from .utils import SwiftAdapter, SwiftConfig, SwiftOutput

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

        def load_state_dict_callback(module: nn.Module, adapter_name: str, state_dict: Dict[str, torch.Tensor]):
            assert adapter_name and '..' not in adapter_name
            adapter_keys = state_dict.keys()
            original_state_dict = {}
            for key, value in module.state_dict().items():
                if key in adapter_keys:
                    original_state_dict[key] = value

            setattr(module, f'{adapter_name}.origin', original_state_dict)
            setattr(module, f'{adapter_name}.adapter', state_dict)

        return SwiftOutput(
            config=config,
            state_dict_callback=state_dict_callback,
            mark_trainable_callback=mark_trainable_callback,
            load_state_dict_callback=load_state_dict_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        if activate:
            state_dict = getattr(module, f'{adapter_name}.adapter', None)
        else:
            state_dict = getattr(module, f'{adapter_name}.origin', None)
        if state_dict:
            incompatible_keys = module.load_state_dict(state_dict, False)
            if incompatible_keys and len(incompatible_keys[1]) > 0:
                logger.error(f'Load state dict with unexpected keys: {incompatible_keys[1]}')
        else:
            logger.warn('No state_dict found on the module for part tuner.')

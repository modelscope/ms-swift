# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from copy import deepcopy
from dataclasses import dataclass
from types import MethodType
from typing import Dict, Optional

import torch
from torch import nn

from swift import get_logger
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
        name_list = [name for name, _ in model.named_modules(remove_duplicate=False)]
        for name in name_list:
            module: nn.Module = model.get_submodule(name)
            if Part.target_module_matched(name, config) and not getattr(module, 'plugin', False):
                if hasattr(module, 'base_layer'):
                    module = module.base_layer

                def _forward(self, *args, **kwargs):
                    child_list = [
                        sub_module for name, sub_module in self.named_modules(remove_duplicate=False)
                        if '_part_' in name
                    ]
                    sub_modules = [child for child in child_list if getattr(child, 'activated', False)]
                    assert len(sub_modules) <= 1
                    if len(sub_modules) == 1:
                        return sub_modules[0].forward(*args, **kwargs)
                    else:
                        return self.forward_origin(*args, **kwargs)

                if not hasattr(module, 'forward_origin'):
                    module.forward_origin = module.forward
                    module.forward = MethodType(_forward, module)

                new_module = deepcopy(module)
                for attr in dir(new_module):
                    if '_part_' in attr:
                        delattr(new_module, attr)
                new_module.part_name = adapter_name
                ActivationMixin.mark_all_sub_modules_as_plugin(new_module)
                setattr(module, f'_part_{adapter_name}', new_module)
                new_module.requires_grad_(True)

        def state_dict_callback(state_dict, adapter_name):
            new_state_dict = {}
            for key, value in state_dict.items():
                if f'_part_{adapter_name}.' in key:
                    new_key = key.replace(f'_part_{adapter_name}.', '').replace('base_layer.', '')
                    new_state_dict[new_key] = value

            return new_state_dict

        def mark_trainable_callback(model: nn.Module):
            pass

        def load_state_dict_callback(model: nn.Module, adapter_name: str, state_dict: Dict[str, torch.Tensor]):
            new_state_dict = {}
            for name, module in model.named_modules(remove_duplicate=False):
                module: nn.Module
                if Part.target_module_matched(name, config):
                    for param_name in state_dict:
                        if param_name.startswith(name):
                            end = param_name[len(name):]
                            if hasattr(module, 'base_layer'):
                                new_state_dict[name + f'.base_layer._part_{adapter_name}'
                                               + end] = state_dict[param_name]
                            else:
                                new_state_dict[name + f'._part_{adapter_name}' + end] = state_dict[param_name]
            return new_state_dict

        return SwiftOutput(
            config=config,
            state_dict_callback=state_dict_callback,
            mark_trainable_callback=mark_trainable_callback,
            load_state_dict_callback=load_state_dict_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        name_list = [name for name, _ in module.named_modules(remove_duplicate=False)]
        for name in name_list:
            sub_module: nn.Module = module.get_submodule(name)
            if re.fullmatch(f'.*_part_{adapter_name}$', name):
                sub_module.activated = activate
                SwiftAdapter.save_memory(sub_module, adapter_name, name, activate, offload)

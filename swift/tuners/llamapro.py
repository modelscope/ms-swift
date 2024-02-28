# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from swift.utils.logger import get_logger
from .module_mapping import MODEL_KEYS_MAPPING
from .utils import SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class LLaMAProConfig(SwiftConfig):
    """
    The configuration class for the LLaMAPro module.

    See https://arxiv.org/abs/2401.02415

    Args:

    """
    num_new_blocks: int = None

    num_groups: Optional[int] = None

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.LLAMAPRO


class LLaMAPro(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: LLaMAProConfig,
                      adapter_name: str) -> SwiftOutput:
        """Prepare a model with `LLaMAProConfig`"""
        num_hidden_layers = model.config.num_hidden_layers
        assert num_hidden_layers % config.num_new_blocks == 0, f'Model layers {num_hidden_layers} ' \
                                                               f'should be divided by {config.num_new_blocks}'
        if config.num_groups is None:
            config.num_groups = config.num_new_blocks

        num_stride = num_hidden_layers // config.num_groups

        # We only support decoder only model for now.
        module_list = LLaMAPro._find_module_list(model)
        new_module_list = []
        is_new_module = []
        for idx, module in module_list:
            if (idx+1) % num_stride == 0:
                new_module = deepcopy(module)
                new_module_list.append(new_module)
                is_new_module.append(True)
            else:
                new_module_list.append(module)
                is_new_module.append(False)

        LLaMAPro._update_module_weight(new_module_list, is_new_module)
        LLaMAPro._update_module_attr(new_module_list, is_new_module)

        def state_dict_callback(state_dict, adapter_name):
            return state_dict

        def mark_trainable_callback(model):
            return

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)

    @staticmethod
    def _update_module_attr(module_list, is_new_module, model_type):
        if model_type == 'llama':
            for idx, module in enumerate(module_list):
                module.self_attn.layer_idx = idx

    @staticmethod
    def _update_module_weight(config: LLaMAProConfig, module_list, is_new_module, model_type):
        for module, is_new in zip(module_list, is_new_module):
            if is_new:
                if model_type in MODEL_KEYS_MAPPING.keys():
                    model_key_mapping = config.model_key_mapping
                else:
                    model_key_mapping = config.model_key_mapping
                    raise f'{model_type} is not defined in MODEL_KEYS_MAPPING, ' \
                          f'please consider pass the information through the config.model_key_mapping'

                if model_type in ('llama', 'mistral', 'qwen2', 'yi', 'gemma'):
                    module.self_attn.o_proj.weight = torch.zeros_like(module.self_attn.o_proj.weight)
                    if hasattr(module.self_attn.o_proj, 'bias'):
                        module.self_attn.o_proj.bias = torch.zeros_like(module.self_attn.o_proj.bias)
                    module.mlp.down_proj.weight = torch.zeros_like(module.mlp.down_proj.weight)
                elif model_type == 'chatglm':
                    module.mlp.dense_4h_to_h.weight = torch.zeros_like(module.mlp.dense_4h_to_h.weight)
                    if hasattr(module.mlp.dense_4h_to_h, 'bias'):
                        module.mlp.dense_4h_to_h.bias = torch.zeros_like(module.mlp.dense_4h_to_h.bias)
                    module.self_attention.dense.weight = torch.zeros_like(module.self_attention.dense.weight)
                    if hasattr(module.self_attention.dense, 'bias'):
                        module.self_attention.dense.bias = torch.zeros_like(module.self_attention.dense.bias)

    @staticmethod
    def _find_module_list(module: nn.Module):
        for sub_module in module.modules():
            if isinstance(sub_module, torch.nn.ModuleList):
                return sub_module

    @staticmethod
    def activate_adapter(module: torch.nn.Module,
                         adapter_name: str,
                         activate: bool,
                         offload: str = None):
        for sub_module in module.modules():
            if isinstance(sub_module, torch.nn.Embedding):
                sub_module.nef_activated = activate

    @staticmethod
    def freeze_model():
        return False

    @staticmethod
    def has_additional_modules():
        return False

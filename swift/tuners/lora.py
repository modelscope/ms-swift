# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
from dataclasses import dataclass, field
from typing import Union

import torch
from peft.tuners.lora import LoraLayer

from swift import LoraConfig
from .lora_layers import *  # noqa
from .utils import SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class LoRAConfig(LoraConfig, SwiftConfig):
    """
    The configuration class for the loRA module.

    Args:
        use_qa_lora(bool): Use
            QA-LoRA:[Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717)
            instead of LoRA. QA-LoRA only supports AutoGPTQ quantized models.
    """

    use_qa_lora: bool = field(
        default=False,
        metadata={
            'help':
            'Use [qa-lora](https://github.com/yuhuixu1993/qa-lora) or not'
        })

    use_merged_linear: bool = field(
        default=False,
        metadata={
            'help': 'Use merged Linear'
        })

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.LORA


class LoRA(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: LoRAConfig, adapter_name: str):
        LoraModel(model, config, adapter_name)

        def state_dict_callback(state_dict, adapter_name):
            return lora_state_dict(state_dict, adapter_name, config.bias)

        def mark_trainable_callback(model):
            mark_lora_as_trainable(model, adapter_name, config.bias)

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str,
                         activate: bool):
        for sub_module in module.modules():
            if isinstance(sub_module, (LoraLayer, LoRALayer)):
                sub_module.set_activation(adapter_name, activate)

    @staticmethod
    def unpatch_lora(model, config: LoRAConfig, adapter_name: Optional[Union[List[str], str]]):
        """Unpatch lora modules and merge the weights to original modules.

        LoRA constructs an additional layer with low-rank decomposition matrices of the weights in the network.
        'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al.(2021)
        See https://arxiv.org/abs/2106.09685

        Args:
            model(`torch.nn.Module`): The model called with `tune` function.
            config(`LoRAConfig`): The `LoRAConfig` to use. Deprecated
            adapter_name(`str`): The adapter name
        """
        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]
        LoraModel(model, None, '').merge_and_unload(adapter_name)

# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
import re
from dataclasses import dataclass, field
from typing import List, Union, Tuple

import torch.nn as nn

from swift import LoRAConfig, LoRA, SwiftOutput
from swift.tuners.lora import mark_lora_as_trainable, lora_state_dict


class LongLoRAModelType:
    LLAMA = 'llama'


@dataclass
class LongLoRAConfig(LoRAConfig):

    embedder_and_normalizer: Union[str, List[str], Tuple[str]] = field(
        default=('embed', 'norm'),
        metadata={
            'help':
            'The names of embedder and normalizer, regex format if is a str, else will match with sub sequences'
        })

    model_type: str = field(
        default=None,
        metadata={
            'help':
                'The model type, now only support `llama` structure.'
        })

    use_flash_attn: bool = field(
        default=False,
        metadata={
            'help':
                'Use flash attention or not.'
        })

    is_trainable: bool = field(
        default=True,
        metadata={
            'help':
                'Use in sft or inference.'
        })

    def __post_init__(self):
        from swift.tuners.mapping import SwiftTuners
        self.swift_type = SwiftTuners.LORA


class LongLoRA(LoRA):

    @staticmethod
    def prepare_model(model: nn.Module, config: LongLoRAConfig, adapter_name: str):
        """Prepare a model with `LoRAConfig`"""
        LoRA._dynamic_patch_lora(
            model,
            target_modules=config.target_modules,
            r=config.r,
            adapter_name=adapter_name,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            merge_weights=config.merge_weights,
            use_merged_linear=config.use_merged_linear,
            enable_lora=config.enable_lora,
            fan_in_fan_out=config.fan_in_fan_out)

        def state_dict_callback(state_dict, adapter_name):
            _state_dict = lora_state_dict(state_dict, adapter_name, config.bias)
            for name, value in state_dict.items():
                if isinstance(config.embedder_and_normalizer, str):
                    target_module_found = re.fullmatch(config.embedder_and_normalizer, name)
                else:
                    target_module_found = any(
                        target_key in name
                        for target_key in config.embedder_and_normalizer)
                if target_module_found and name not in _state_dict:  # noqa
                    _state_dict[name] = value
            return _state_dict

        def mark_trainable_callback(model):
            mark_lora_as_trainable(model, adapter_name, config.bias)
            mark_embedding_normalizer_as_trainable(model, config.embedder_and_normalizer)

        if config.model_type == LongLoRAModelType.LLAMA:
            from .llama import replace_llama_attn
            replace_llama_attn(use_flash_attn=config.use_flash_attn, inference=config.is_trainable)

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)


def mark_embedding_normalizer_as_trainable(model: nn.Module, extra_parameters: Union[str, List[str], Tuple[str]]) -> None:
    for name, sub_module in model.named_parameters():
        if isinstance(extra_parameters, str):
            target_module_found = re.fullmatch(extra_parameters, name)
        else:
            target_module_found = any(
                target_key in name
                for target_key in extra_parameters)
        if target_module_found:  # noqa
            sub_module.requires_grad = True







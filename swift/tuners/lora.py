# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
import math
import re
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict, List, Union

import torch
import torch.nn as nn
from loralib import Conv1d as _Conv1d
from loralib import Conv2d as _Conv2d
from loralib import Conv3d as _Conv3d
from loralib import Embedding as _Embedding
from loralib import Linear as _Linear
from loralib import MergedLinear as _MergedLinear
from peft.import_utils import (is_auto_gptq_available, is_bnb_4bit_available,
                               is_bnb_available)
from peft.utils import get_auto_gptq_quant_linear, get_quantization_config

from swift import get_logger
from ..utils.torch_utils import find_sub_module
from .utils import ActivationMixin, SwiftConfig, SwiftOutput


class Linear(ActivationMixin, _Linear):

    def __init__(self, *args, **kwargs):
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(Linear, self).__init__()
        if not self._unique_thread:
            self.merge_weights = False

    def forward(self, x: torch.Tensor, **kwargs):
        return super().forward(x)


class MergedLinear(ActivationMixin, _MergedLinear):

    def __init__(self, *args, **kwargs):
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(MergedLinear, self).__init__()
        if not self._unique_thread:
            self.merge_weights = False

    def forward(self, x: torch.Tensor, **kwargs):
        return super().forward(x)


class Embedding(ActivationMixin, _Embedding):

    def __init__(self, *args, **kwargs):
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(Embedding, self).__init__()
        if not self._unique_thread:
            self.merge_weights = False

    def forward(self, x: torch.Tensor, **kwargs):
        return super().forward(x)


class Conv2d(ActivationMixin, _Conv2d):

    def __init__(self, *args, **kwargs):
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(Conv2d, self).__init__()
        if not self._unique_thread:
            self.merge_weights = False

    def forward(self, x: torch.Tensor, **kwargs):
        return super().forward(x)


class Conv1d(ActivationMixin, _Conv1d):

    def __init__(self, *args, **kwargs):
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(Conv1d, self).__init__()
        if not self._unique_thread:
            self.merge_weights = False

    def forward(self, x: torch.Tensor, **kwargs):
        return super().forward(x)


class Conv3d(ActivationMixin, _Conv3d):

    def __init__(self, *args, **kwargs):
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(Conv3d, self).__init__()
        if not self._unique_thread:
            self.merge_weights = False

    def forward(self, x: torch.Tensor, **kwargs):
        return super().forward(x)


if is_bnb_available():
    import bitsandbytes as bnb

    from peft.tuners.lora import Linear8bitLt as _Linear8bitLt

    class Linear8bitLt(ActivationMixin, _Linear8bitLt):

        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            super(ActivationMixin,
                  self).__init__(adapter_name, in_features, out_features, r,
                                 lora_alpha, lora_dropout, **kwargs)
            super(Linear8bitLt, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.is_activated():
                return bnb.nn.Linear8bitLt.forward(self, x)
            return super().forward(x)


if is_bnb_4bit_available():
    from peft.tuners.lora import Linear4bit as _Linear4bit

    class Linear4bit(ActivationMixin, _Linear4bit):

        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            super(ActivationMixin,
                  self).__init__(adapter_name, in_features, out_features, r,
                                 lora_alpha, lora_dropout, **kwargs)
            super(Linear4bit, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.is_activated():
                return bnb.nn.Linear4bit.forward(self, x)
            return super().forward(x)


if is_auto_gptq_available():
    from peft.tuners.lora import QuantLinear as _QuantLinear

    class QuantLinear(ActivationMixin, _QuantLinear):

        def __init__(
            self,
            adapter_name,
            quant_linear_module,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            super(ActivationMixin,
                  self).__init__(adapter_name, quant_linear_module, r,
                                 lora_alpha, lora_dropout, **kwargs)
            super(QuantLinear, self).__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.is_activated():
                return self.quant_linear_module(x)
            return super().forward(x)


logger = get_logger()


@dataclass
class LoRAConfig(SwiftConfig):
    """
    The configuration class for the loRA module.

    Args:
        r(int): The rank of the LoRA module
        target_modules(List[str]): The modules to be replaced by LoRA,
            can be the end of the module name or a regex string
        lora_alpha(float): The factor to add the lora weights
        lora_dropout(float): The dropout rate of the lora module
        merge_weights(bool): Whether to merge weights when validating
        use_merged_linear(bool): Whether to replace with merged linear layer
        enable_lora(List[bool]): The modules need to be turned on when using the merged linear layer
        fan_in_fan_out(bool): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        bias(str): Bias type. Values ca be "none", "all" or "lora_only"
    """

    r: int = field(default=6, metadata={'help': 'The rank of the LoRA module'})
    target_modules: List[str] = field(
        default=None,
        metadata={
            'help':
            'The modules to be replaced by LoRA, can be the end of the module name or a regex string'
        })
    lora_alpha: float = field(
        default=1., metadata={'help': 'The factor to add the lora weights'})
    lora_dropout: float = field(
        default=0., metadata={'help': 'The dropout rate of the lora module'})
    merge_weights: bool = field(
        default=True,
        metadata={'help': 'Whether to merge weights when validating'})
    use_merged_linear: bool = field(
        default=False,
        metadata={'help': 'Whether to replace with merged linear layer'})
    enable_lora: List[bool] = field(
        default=None,
        metadata={
            'help':
            'The modules need to be turned on when using the merged linear layer'
        })
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            'help':
            'Set this to True if the layer to replace stores weight like (fan_in, fan_out)'
        })
    bias: str = field(
        default='none',
        metadata={
            'help': 'Bias type. Values ca be "none", "all" or "lora_only"'
        })

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.LORA


class LoRA:

    @staticmethod
    def prepare_model(model: nn.Module, config: LoRAConfig, adapter_name: str):
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
            return lora_state_dict(state_dict, adapter_name, config.bias)

        def mark_trainable_callback(model):
            mark_lora_as_trainable(model, adapter_name, config.bias)

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str,
                         activate: bool):
        modules: List[torch.nn.Module] = find_sub_module(
            module, f'loramodule_{adapter_name}')
        for _module in modules:
            _module: ActivationMixin
            _module.set_activation(activate)

    @staticmethod
    def _dynamic_patch_lora(model: torch.nn.Module,
                            target_modules: Union[str, List[str]],
                            use_merged_linear: bool, adapter_name: str,
                            **kwargs):
        """Dynamic patch lora to model

        Args:
            model(`torch.nn.Module`): The torch.nn.Module containing the target module to be patched.
            target_modules(`Union[str, List[str]]`): The module names to be replaced,
                the replacing strategy is `end with`.
            use_merged_linear(bool): Whether to replace with merged linear layer.
            adapter_name(str): The adapter name.
            **kwargs: The arguments passed from `tune` which are needed by lora.
        """
        modules = {}
        module_keys = [key for key, _ in model.named_modules()]
        assert isinstance(target_modules, (str, list))
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(
            get_quantization_config(model, method='gptq'))

        for module_key in module_keys:
            if isinstance(target_modules, str):
                target_module_found = re.fullmatch(target_modules, module_key)
            else:
                target_module_found = any(
                    module_key.endswith(target_key)
                    for target_key in target_modules)
            if target_module_found:  # noqa
                sub_module = model.get_submodule(module_key)

                lora_module = None
                if getattr(model, 'is_loaded_in_8bit', False) and isinstance(
                        sub_module, bnb.nn.Linear8bitLt):
                    eight_bit_kwargs = kwargs.copy()
                    eight_bit_kwargs.update({
                        'has_fp16_weights':
                        sub_module.state.has_fp16_weights,
                        'memory_efficient_backward':
                        sub_module.state.memory_efficient_backward,
                        'threshold':
                        sub_module.state.threshold,
                        'index':
                        sub_module.index,
                    })
                    lora_module = Linear8bitLt(
                        'default',
                        sub_module.in_features,
                        sub_module.out_features,
                        bias=hasattr(sub_module, 'bias')
                        and sub_module.bias is not None,
                        **eight_bit_kwargs)
                elif getattr(model, 'is_loaded_in_4bit',
                             False) and is_bnb_4bit_available() and isinstance(
                                 sub_module, bnb.nn.Linear4bit):
                    four_bit_kwargs = kwargs.copy()
                    four_bit_kwargs.update({
                        'compute_dtype':
                        sub_module.compute_dtype,
                        'compress_statistics':
                        sub_module.weight.compress_statistics,
                        'quant_type':
                        sub_module.weight.quant_type,
                    })
                    lora_module = Linear4bit(
                        'default',
                        sub_module.in_features,
                        sub_module.out_features,
                        bias=hasattr(sub_module, 'bias')
                        and sub_module.bias is not None,
                        **four_bit_kwargs)
                elif AutoGPTQQuantLinear is not None and isinstance(
                        sub_module, AutoGPTQQuantLinear):
                    lora_module = QuantLinear('default', sub_module, **kwargs)
                    sub_module.weight = sub_module.qweight
                elif isinstance(sub_module, torch.nn.Linear):
                    if use_merged_linear:
                        lora_module = MergedLinear(
                            sub_module.in_features,
                            sub_module.out_features,
                            bias=hasattr(sub_module, 'bias')
                            and sub_module.bias is not None,
                            **kwargs)
                    else:
                        kwargs.pop('enable_lora', None)
                        lora_module = Linear(
                            sub_module.in_features,
                            sub_module.out_features,
                            bias=hasattr(sub_module, 'bias')
                            and sub_module.bias is not None,
                            **kwargs)
                elif isinstance(sub_module, torch.nn.Embedding):
                    lora_module = Embedding(
                        num_embeddings=sub_module.num_embeddings,
                        embedding_dim=sub_module.embedding_dim,
                        padding_idx=sub_module.padding_idx,
                        max_norm=sub_module.max_norm,
                        norm_type=sub_module.norm_type,
                        scale_grad_by_freq=sub_module.scale_grad_by_freq,
                        sparse=sub_module.sparse,
                        r=kwargs['r'],
                        lora_alpha=kwargs['lora_alpha'],
                        merge_weights=kwargs['merge_weights'],
                    )
                elif isinstance(sub_module, torch.nn.Conv2d):
                    kwargs.pop('fan_in_fan_out', None)
                    lora_module = Conv2d(
                        sub_module.in_channels,
                        sub_module.out_channels,
                        kernel_size=sub_module.kernel_size,
                        stride=sub_module.stride,
                        padding=sub_module.padding,
                        dilation=sub_module.dilation,
                        groups=sub_module.groups,
                        **kwargs)

                def _forward(self, *args, **kwargs):
                    for _name, _module in self.named_modules():
                        if 'loramodule_' in _name and _module.is_activated():
                            return _module.forward(*args, **kwargs)
                    return self.forward_origin(*args, **kwargs)

                if lora_module is not None:
                    lora_module.weight = sub_module.weight
                    if getattr(sub_module, 'bias', None) is not None:
                        lora_module.bias = sub_module.bias
                    if getattr(sub_module, 'state', None) is not None:
                        lora_module.state = sub_module.state
                    lora_module.to(sub_module.weight.device)
                    setattr(sub_module, f'loramodule_{adapter_name}',
                            lora_module)
                    if not hasattr(sub_module, 'forward_origin'):
                        sub_module.forward_origin = sub_module.forward
                        sub_module.forward = MethodType(_forward, sub_module)
                    modules[module_key] = adapter_name

        logger.debug(f'Lora modules(module_key -> adapter_name): {modules}')

    @staticmethod
    def unpatch_lora(model, config: LoRAConfig, adapter_name: str):
        """Unpatch lora modules and merge the weights to original modules.

        LoRA constructs an additional layer with low-rank decomposition matrices of the weights in the network.
        'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al.(2021)
        See https://arxiv.org/abs/2106.09685

        Args:
            model(`torch.nn.Module`): The model called with `tune` function.
            config(`LoRAConfig`): The `LoRAConfig` to use.
            adapter_name(`str`): The adapter name
        """
        module_keys = [key for key, _ in model.named_modules()]
        assert isinstance(config.target_modules, (str, list))
        target_modules = config.target_modules

        for module_key in module_keys:
            if isinstance(target_modules, str):
                target_module_found = re.fullmatch(target_modules, module_key)
            else:
                target_module_found = any(
                    module_key.endswith(target_key)
                    for target_key in target_modules)
            if target_module_found:  # noqa
                sub_module = model.get_submodule(module_key)
                lora_module = getattr(sub_module, f'loramodule_{adapter_name}')

                if lora_module is not None:
                    if hasattr(lora_module, 'merge_weights'):
                        lora_module.merge_weights = True
                        lora_module.eval()
                        delattr(sub_module, f'loramodule_{adapter_name}')


def mark_lora_as_trainable(model: nn.Module,
                           adapter_name: str,
                           bias: str = 'none') -> None:
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for n, m in model.named_modules():
            if f'loramodule_{adapter_name}' in n and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(state_dict,
                    adapter_name: str,
                    bias: str = 'none') -> Dict[str, torch.Tensor]:
    if bias == 'none':
        return {
            k: state_dict[k]
            for k in state_dict
            if f'loramodule_{adapter_name}' in k and 'lora_' in k
        }
    elif bias == 'all':
        return {
            k: state_dict[k]
            for k in state_dict
            if ('lora_' in k and f'loramodule_{adapter_name}' in k) or (
                'bias' in k and f'loramodule_{adapter_name}' not in k)
        }
    elif bias == 'lora_only':
        to_return = {}
        for k in state_dict:
            if f'loramodule_{adapter_name}' in k and 'lora_' in k:
                to_return[k] = state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

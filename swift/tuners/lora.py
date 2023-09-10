# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.import_utils import (is_auto_gptq_available, is_bnb_4bit_available,
                               is_bnb_available)
from peft.utils import get_auto_gptq_quant_linear, get_quantization_config

from ..utils.torch_utils import find_sub_module
from .utils import SwiftConfig, SwiftOutput

if is_bnb_available():
    import bitsandbytes as bnb

    from peft.tuners.lora import Linear8bitLt

if is_bnb_4bit_available():
    from peft.tuners.lora import Linear4bit

if is_auto_gptq_available():
    from peft.tuners.lora import QuantLinear

logger = logging.getLogger()


@dataclass
class LoRAConfig(SwiftConfig):
    """
    The configuration class for the loRA module.

    Args:
        r: The rank of the LoRA module
        target_modules: The modules to be replaced by LoRA, can be the end of the module name or a regex string
        lora_alpha: The factor to add the lora weights
        lora_dropout: The dropout rate of the lora module
        merge_weights: Whether to merge weights when validating
        use_merged_linear: Whether to replace with merged linear layer
        enable_lora: The modules need to be turned on when using the merged linear layer
        fan_in_fan_out: Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        bias: Bias type. Values ca be "none", "all" or "lora_only"
    """

    r: int = field(default=6, metadata={'help': 'The rank of the LoRA module'})
    target_modules: List = field(
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
            replace_modules=config.target_modules,
            r=config.r,
            adapter_name=adapter_name,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            merge_weights=config.merge_weights,
            use_merged_linear=config.use_merged_linear,
            enable_lora=config.enable_lora,
            fan_in_fan_out=config.fan_in_fan_out)

        def state_dict_callback(state_dict, adapter_name):
            return lora_state_dict(state_dict, model.lora_module_map,
                                   adapter_name, config.bias)

        def mark_trainable_callback(model):
            mark_lora_as_trainable(model, config.bias)

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str,
                         activate: bool):
        modules: List[torch.nn.Module] = find_sub_module(module, adapter_name)
        for _module in modules:
            if isinstance(_module, LoRALayer):
                _module.activate(activate)
            else:
                _module.active_adapter = 'default' if activate else 'invalid'

    @staticmethod
    def _dynamic_patch_lora(model, replace_modules, use_merged_linear,
                            adapter_name, **kwargs):
        """Dynamic patch lora to model

        Args:
            model: The torch.nn.Module containing the target module to be patched.
            replace_modules: The module names to be replaced, the replacing strategy is `end with`.
            use_merged_linear: Whether to replace with merged linear layer
            **kwargs: The arguments passed from `tune` which are needed by lora.

        Returns:
            The lora modules
        """
        if not hasattr(model, 'lora_module_map'):
            model.lora_module_map = {}
        modules = {}
        module_keys = [key for key, _ in model.named_modules()]
        assert isinstance(replace_modules, (str, list))
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(
            get_quantization_config(model, method='gptq'))

        for module_key in module_keys:
            if isinstance(replace_modules, str):
                target_module_found = re.fullmatch(replace_modules, module_key)
            else:
                target_module_found = any(
                    module_key.endswith(target_key)
                    for target_key in replace_modules)
            if target_module_found:  # noqa
                parts = module_key.split('.')
                module = model.get_submodule('.'.join(parts[:-1]))
                sub_module = model.get_submodule(module_key)
                _key = parts[-1]

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

                if lora_module is not None:
                    lora_module.weight = sub_module.weight
                    if getattr(sub_module, 'bias', None) is not None:
                        lora_module.bias = sub_module.bias
                    if getattr(sub_module, 'state', None) is not None:
                        lora_module.state = sub_module.state
                    lora_module.to(sub_module.weight.device)
                    lora_module.adapter_name = adapter_name
                    setattr(module, _key, lora_module)
                    modules[module_key] = adapter_name

        model.lora_module_map.update(modules)

    @staticmethod
    def unpatch_lora(model, config: LoRAConfig):
        """Unpatch lora modules and merge the weights to original modules.

        LoRA constructs an additional layer with low-rank decomposition matrices of the weights in the network.
        'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al.(2021)
        See https://arxiv.org/abs/2106.09685

        Args:
            model: The model called with `tune` function.
            config: The `LoRAConfig` to use.
        """
        if not hasattr(model, 'lora_module_map'):
            model.lora_module_map = {}
        module_keys = [key for key, _ in model.named_modules()]
        assert isinstance(config.replace_modules, (str, list))
        replace_modules = config.replace_modules

        for module_key in module_keys:
            if isinstance(replace_modules, str):
                target_module_found = re.fullmatch(replace_modules, module_key)
            else:
                target_module_found = any(
                    module_key.endswith(target_key)
                    for target_key in replace_modules)
            if target_module_found:  # noqa
                parts = module_key.split('.')
                module = model.get_submodule('.'.join(parts[:-1]))
                sub_module = model.get_submodule(module_key)
                _key = parts[-1]

                origin_module = None
                if isinstance(sub_module, Linear):
                    origin_module = torch.nn.Linear(
                        sub_module.in_features,
                        sub_module.out_features,
                        bias=hasattr(sub_module, 'bias')
                        and sub_module.bias is not None,
                    )
                elif isinstance(sub_module, Embedding):
                    origin_module = torch.nn.Embedding(
                        num_embeddings=sub_module.num_embeddings,
                        embedding_dim=sub_module.embedding_dim,
                        padding_idx=sub_module.padding_idx,
                        max_norm=sub_module.max_norm,
                        norm_type=sub_module.norm_type,
                        scale_grad_by_freq=sub_module.scale_grad_by_freq,
                        sparse=sub_module.sparse,
                    )
                elif isinstance(sub_module, Conv2d):
                    origin_module = torch.nn.Conv2d(
                        sub_module.in_channels,
                        sub_module.out_channels,
                        kernel_size=sub_module.kernel_size,
                        stride=sub_module.stride,
                        padding=sub_module.padding,
                        dilation=sub_module.dilation,
                        groups=sub_module.groups)

                if origin_module is not None:
                    sub_module.merge_weights = True
                    sub_module.eval()
                    origin_module.weight = sub_module.weight
                    if getattr(sub_module, 'bias', None) is not None:
                        origin_module.bias = sub_module.bias
                    origin_module.to(sub_module.weight.device).to(
                        sub_module.weight.dtype)
                    setattr(module, _key, origin_module)
                    model.lora_module_map.pop(module_key, None)


class LoRALayer:

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.old_r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

    def activate(self, activate=True):
        if activate:
            self.r = self.old_r
        else:
            self.r = 0


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 merge_weights: bool = True,
                 **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        self.lora_A.requires_grad = mode
        self.lora_B.requires_grad = mode
        if mode and self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B
                                     @ self.lora_A).T * self.scaling
            self.merged = False
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B
                                     @ self.lora_A).T * self.scaling
            self.merged = True

    def eval(self):
        nn.Embedding.eval(self)
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(x, self.lora_A.T, self.padding_idx,
                                      self.max_norm, self.norm_type,
                                      self.scale_grad_by_freq, self.sparse)
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        self.lora_A.requires_grad = mode
        self.lora_B.requires_grad = mode
        if mode and self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                x_dtype = x.dtype
                x = x.to(self.lora_A.dtype)
                result += (self.lora_dropout(x) @ self.lora_A.T
                           @ self.lora_B.T) * self.scaling
                result = result.to(x_dtype)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.,
                 enable_lora: List[bool] = [False],
                 fan_in_fan_out: bool = False,
                 merge_weights: bool = True,
                 **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_features // len(enable_lora) * sum(enable_lora),
                     r)))  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1,
            self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        self.lora_A.requires_grad = mode
        self.lora_B.requires_grad = mode
        if mode and self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0),
                    self.lora_B.data.unsqueeze(-1),
                    groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False
        if not mode and self.merge_weights and not self.merged:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0),
                    self.lora_B.data.unsqueeze(-1),
                    groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0),
                    self.lora_B.data.unsqueeze(-1),
                    groups=sum(self.enable_lora)).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                x_dtype = x.dtype
                x = x.to(self.lora_A.dtype)
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1),
                    self.lora_B.unsqueeze(-1),
                    groups=sum(self.enable_lora)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
                result = result.to(x_dtype)
            return result


class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.,
                 merge_weights: bool = True,
                 **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                           **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros(
                    (r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        self.lora_A.requires_grad = mode
        self.lora_B.requires_grad = mode
        if mode and self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(
                self.weight.shape) * self.scaling
            self.merged = False
        if not mode and self.merge_weights and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).view(
                self.weight.shape) * self.scaling
            self.merged = True

    def eval(self):
        nn.Conv2d.eval(self)
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(
                self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight +  # noqa
                (self.lora_B @ self.lora_A).view(self.weight.shape)  # noqa
                * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups)
        return nn.Conv2d.forward(self, x)


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
        for m in model.modules():
            if adapter_name == getattr(m, 'adapter_name', None) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(state_dict,
                    module_map: Dict,
                    adapter_name: str,
                    bias: str = 'none') -> Dict[str, torch.Tensor]:
    if bias == 'none':
        return {
            k: state_dict[k]
            for k in state_dict
            if 'lora_' in k and module_map.get(k[:k.find('lora_')
                                                 - 1], None) == adapter_name
        }
    elif bias == 'all':
        return {
            k: state_dict[k]
            for k in state_dict if ('lora_' in k and module_map.get(
                k[:k.find('lora_') - 1], None) == adapter_name) or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        for k in state_dict:
            if 'lora_' in k and module_map.get(k[:k.find('lora_') - 1],
                                               None) == adapter_name:
                to_return[k] = state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

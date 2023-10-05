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
import torch.nn.functional as F
from peft.import_utils import (is_auto_gptq_available, is_bnb_4bit_available,
                               is_bnb_available)
from peft.utils import get_auto_gptq_quant_linear, get_quantization_config

from swift import get_logger
from ..utils.torch_utils import find_sub_module
from .utils import ActivationMixin, SwiftConfig, SwiftOutput

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
            use_qa_lora: bool = False,
            **kwargs,
        ):
            super(ActivationMixin,
                  self).__init__(adapter_name, quant_linear_module, r,
                                 lora_alpha, lora_dropout, **kwargs)
            super(QuantLinear, self).__init__()
            self.use_qa_lora = use_qa_lora
            if self.use_qa_lora:
                assert 'group_size' in kwargs, 'To use qa_lora you need to pass in the `group_size` param.'
                self.qa_pool = torch.nn.AvgPool1d(
                    kwargs['group_size']
                )  # using pooling layer to conduct sum operation

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = self.quant_linear_module(x)
            if not self.is_activated(
            ) or self.disable_adapters or self.active_adapter not in self.lora_A.keys(
            ):
                return result
            elif self.r[self.active_adapter] > 0:
                result = result.clone()
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype
                    x = x.to(self.lora_A[self.active_adapter].weight.dtype)
                    if self.use_qa_lora:
                        x = self.qa_pool(x)
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](self.lora_dropout[
                                self.active_adapter](x))).to(expected_dtype)
                        * self.scaling[self.active_adapter])
                else:
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](
                                self.lora_dropout[self.active_adapter](x)))
                        * self.scaling[self.active_adapter])
                result += output
            return result


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

    use_qa_lora: bool = field(
        default=False,
        metadata={
            'help':
            'Use [qa-lora](https://github.com/yuhuixu1993/qa-lora) or not'
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
            fan_in_fan_out=config.fan_in_fan_out,
            use_qa_lora=config.use_qa_lora)

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
        auto_gptq_config = get_quantization_config(model, method='gptq')
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(auto_gptq_config)
        use_qa_lora = kwargs.pop('use_qa_lora', False)

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
                    lora_module = QuantLinear(
                        'default',
                        sub_module,
                        use_qa_lora=use_qa_lora,
                        group_size=getattr(auto_gptq_config, 'group_size',
                                           None),
                        **kwargs)
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


class LoRALayer(ActivationMixin):

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        if not self._unique_thread:
            self.merge_weights = False


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
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(
                        0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(
                        0, 1) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor, **kwargs):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            x_dtype = x.dtype
            x = x.to(self.lora_A.dtype)
            after_A = F.embedding(x, self.lora_A.transpose(0, 1),
                                  self.padding_idx, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq,
                                  self.sparse)
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            result = result.to(x_dtype)
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
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(
                        self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(
                        self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor, **kwargs):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            x_dtype = x.dtype
            x = x.to(self.lora_A.dtype)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                       @ self.lora_B.transpose(0, 1)) * self.scaling
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
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor, **kwargs):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                x_dtype = x.dtype
                x = x.to(self.lora_A.dtype)
                result += self.lora_dropout(x) @ T(
                    self.merge_AB().T) * self.scaling
                result = result.to(x_dtype)
            return result


class ConvLoRA(nn.Module, LoRALayer):

    def __init__(self,
                 conv_module,
                 in_channels,
                 out_channels,
                 kernel_size,
                 r=0,
                 lora_alpha=1,
                 lora_dropout=0.,
                 merge_weights=True,
                 **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size,
                                **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros(
                    (r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros(
                    (out_channels // self.conv.groups * kernel_size,
                     r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(
                        self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(
                        self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x, **kwargs):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, self.conv.weight +
                (self.lora_B @ self.lora_A).view(self.conv.weight.shape)
                * self.scaling, self.conv.bias)
        return self.conv(x)


class Conv2d(ConvLoRA):

    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):

    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this


class Conv3d(ConvLoRA):

    def __init__(self, *args, **kwargs):
        super().__init__(nn.Conv3d, *args, **kwargs)


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

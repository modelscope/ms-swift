# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.import_utils import (is_auto_gptq_available, is_bnb_4bit_available,
                               is_bnb_available)
from peft.tuners.lora import Embedding as _Embedding, Linear as _Linear, Conv2d as _Conv2d

from swift import get_logger
from .utils import ActivationMixin

logger = get_logger()


class LoRAActivationMixin(ActivationMixin):

    @property
    def active_adapters(self):
        return self.get_activated_adapters()

    @property
    def active_adapter(self) -> str:
        return self.get_activated_adapters()

    def set_adapter(self, adapter_names):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    self.set_activation(key, True)
                    layer.requires_grad_(True)
                else:
                    self.set_activation(key, False)
                    layer.requires_grad_(False)

    def merge(self, *args, **kwargs):
        if not self.unique_thread:
            raise AssertionError('Merge is unsupported in multi thread scenario!')
        return super().merge(*args, **kwargs)



if is_bnb_available():
    from peft.tuners.lora import Linear8bitLt as _Linear8bitLt

    class Linear8bitLt(LoRAActivationMixin, _Linear8bitLt):

        def __init__(
            self,
            adapter_name,
            base_layer,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            super(ActivationMixin,
                  self).__init__(adapter_name, base_layer,
                                 r, lora_alpha, lora_dropout, **kwargs)
            super(Linear8bitLt, self).__init__()


if is_bnb_4bit_available():
    from peft.tuners.lora import Linear4bit as _Linear4bit

    class Linear4bit(LoRAActivationMixin, _Linear4bit):

        def __init__(
            self,
            adapter_name,
            base_layer,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            super(ActivationMixin,
                  self).__init__(adapter_name, base_layer,
                                 r, lora_alpha, lora_dropout, **kwargs)
            super(Linear4bit, self).__init__()


if is_auto_gptq_available():
    from peft.tuners.lora import QuantLinear as _QuantLinear

    class QuantLinear(LoRAActivationMixin, _QuantLinear):

        def __init__(
            self,
            *args,
            use_qa_lora=False,
            group_size=None,
            **kwargs,
        ):
            super(ActivationMixin,
                  self).__init__(*args, **kwargs)
            super(QuantLinear, self).__init__()
            self.group_size = group_size
            self.use_qa_lora = use_qa_lora
            if self.use_qa_lora:
                assert self.group_size is not None, 'To use qa_lora you need to pass in the `group_size` param.'
            if self.use_qa_lora:
                self.qa_pool = torch.nn.AvgPool1d(
                    self.group_size
                )  # using pooling layer to conduct sum operation

        def forward(self, x: torch.Tensor):
            # note: logic differs from default Linear because merging is not supported
            result = self.quant_linear_module(x)

            if self.disable_adapters:
                return result

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                if self.use_qa_lora:
                    x = self.qa_pool(x) * self.group_size
                output = lora_B(lora_A(dropout(x)))
                if requires_conversion:
                    output = output.to(expected_dtype)
                output = output * scaling
                result += output
            return result


class Embedding(LoRAActivationMixin, _Embedding):

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(Embedding, self).__init__()


class Linear(LoRAActivationMixin, _Linear):

    def __init__(
            self,
            *args,
            **kwargs):
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(Linear, self).__init__()


class Conv2d(LoRAActivationMixin, _Conv2d):

    def __init__(
            self,
            *args,
            **kwargs):
        super(ActivationMixin, self).__init__(*args, **kwargs)
        super(Conv2d, self).__init__()


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

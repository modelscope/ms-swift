# Copyright (c) Alibaba, Inc. and its affiliates.
# Code borrowed from huggingface/peft
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (TEColumnParallelLinear, TELayerNormColumnParallelLinear,
                                                         TELinear, TERowParallelLinear)
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from peft.tuners.lora import model
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer


class LoraParallelLinear(MegatronModule, LoraLayer):

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ):
        config = base_layer.config
        super().__init__(config=config)
        LoraLayer.__init__(self, base_layer=base_layer)

        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA yet, please set it to False')

        self.is_parallel_a = isinstance(base_layer, TERowParallelLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.tp_size = base_layer.tp_size
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
        )

        self.is_target_conv_1d_layer = False

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        lora_bias,
    ):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        # lora needs to be forced to upgrade to 32-bit precision, otherwise it will overflow
        origin_params_dtype = self.config.params_dtype
        self.config.params_dtype = torch.float32
        if self.is_parallel_a:
            self.in_features = self.in_features * self.tp_size
            lora_a = TERowParallelLinear(
                input_size=self.in_features,
                output_size=r,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                init_method=self.config.init_method,
                config=self.config,
                is_expert=False,  # TODO: fix MoE
            )
            lora_b = nn.Linear(in_features=r, out_features=self.out_features, bias=lora_bias, dtype=torch.float32)
        else:
            self.out_features = self.out_features * self.tp_size
            lora_a = nn.Linear(in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32)
            lora_b = TEColumnParallelLinear(
                input_size=r,
                output_size=self.out_features,
                bias=lora_bias,
                gather_output=False,
                skip_bias_add=False,
                init_method=self.config.init_method,
                config=self.config,
                is_expert=False,  # TODO: fix MoE
            )
        self.config.params_dtype = origin_params_dtype
        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        self.lora_bias[adapter_name] = lora_bias
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / (r**0.5)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(self.get_base_layer(), 'weight', None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        # If weight is used for matrix multiplication here, the final aggregation operation of the original
        # parallel_linear layer will be missing, so we need to directly call its forward function to obtain the
        # output of the original parallel_linear layer.
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result, bias = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result, bias = self.base_layer(x, *args, **kwargs)
        else:
            if isinstance(self.base_layer, TELayerNormColumnParallelLinear):
                self.base_layer.return_layernorm_output = True
                result, bias = self.base_layer(x, *args, **kwargs)
                result, x = result  # ln_out
            elif isinstance(self.base_layer, TERowParallelLinear):
                result, bias = self.base_layer(x, *args, **kwargs)
            else:
                raise ValueError(f'Unsupported base layer type: {type(self.base_layer)}')
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                lora_result = lora_A(dropout(x))
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_B(lora_result)
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_result * scaling

                result = result + lora_result

        result = result.to(previous_dtype)
        return result, bias

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        res = super().sharded_state_dict(prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata)
        if prefix.endswith('.linear_fc1.'):
            res[f'{prefix}base_layer.weight'] = apply_swiglu_sharded_factory(res[f'{prefix}base_layer.weight'],
                                                                             sharded_offsets)
        return res


def dispatch_megatron(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    linear_cls = (TELayerNormColumnParallelLinear, TELinear)
    if isinstance(target_base_layer, linear_cls):
        new_module = LoraParallelLinear(base_layer=target, adapter_name=adapter_name, **kwargs)

    return new_module


model.dispatch_megatron = dispatch_megatron

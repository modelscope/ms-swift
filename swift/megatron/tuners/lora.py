# Copyright (c) Alibaba, Inc. and its affiliates.
# Code borrowed from huggingface/peft
import math
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

import megatron.core
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import (TEColumnParallelGroupedLinear, TEColumnParallelLinear,
                                                         TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear,
                                                         TERowParallelGroupedLinear, TERowParallelLinear)
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.parallel_state import get_expert_tensor_parallel_world_size, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.router import TopKRouter
from packaging import version
from peft.tuners.lora import model
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from swift.utils import get_current_device
from ..utils import tuners_sharded_state_dict

megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


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

        self.is_parallel_a = isinstance(base_layer, (TERowParallelLinear, TERowParallelGroupedLinear))
        self.is_grouped = isinstance(base_layer, TEGroupedLinear)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.is_expert = getattr(base_layer, 'is_expert', False)
        self.sequence_parallel = getattr(base_layer, 'sequence_parallel', False)
        if self.is_expert:
            self.tp_size = get_expert_tensor_parallel_world_size()
        else:
            self.tp_size = get_tensor_model_parallel_world_size()
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

    def update_layer(self, adapter_name, r, *, lora_alpha, lora_dropout, init_lora_weights, use_rslora, lora_bias,
                     **kwargs):
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
        kwargs = {
            'skip_bias_add': False,
            'init_method': self.config.init_method,
            'config': self.config,
            'is_expert': self.is_expert,
        }
        if megatron_core_013:
            kwargs['tp_group'] = self.base_layer.tp_group
        if isinstance(self.base_layer, TopKRouter):
            router_shape = self.base_layer.weight.shape
            lora_a = TELinear(
                input_size=router_shape[1],
                output_size=r,
                bias=lora_bias,
                parallel_mode=None,
                skip_weight_param_allocation=False,
                **kwargs,
            )
            lora_b = TELinear(
                input_size=r,
                output_size=router_shape[0],
                bias=lora_bias,
                parallel_mode=None,
                skip_weight_param_allocation=False,
                **kwargs,
            )
        elif self.is_parallel_a:
            in_features = self.in_features * self.tp_size
            if self.is_grouped:
                lora_a = TERowParallelGroupedLinear(
                    num_gemms=self.base_layer.num_gemms,
                    input_size=in_features,
                    output_size=r,
                    bias=False,
                    **kwargs,
                )
                lora_b = TEGroupedLinear(
                    num_gemms=self.base_layer.num_gemms,
                    input_size=r,
                    output_size=self.out_features,
                    bias=lora_bias,
                    parallel_mode=None,
                    **kwargs,
                )
            else:
                lora_a = TERowParallelLinear(
                    input_size=in_features,
                    output_size=r,
                    bias=False,
                    input_is_parallel=True,
                    **kwargs,
                )
                lora_b = TELinear(
                    input_size=r,
                    output_size=self.out_features,
                    bias=lora_bias,
                    parallel_mode=None,
                    skip_weight_param_allocation=False,
                    **kwargs,
                )
                lora_a.parallel_mode = self.base_layer.parallel_mode  # fix moe_shared_expert_overlap
        else:
            out_features = self.out_features * self.tp_size
            if self.is_grouped:
                lora_a = TEGroupedLinear(
                    num_gemms=self.base_layer.num_gemms,
                    input_size=self.in_features,
                    output_size=r,
                    bias=lora_bias,
                    parallel_mode=None,
                    **kwargs)
                lora_b = TEColumnParallelGroupedLinear(
                    num_gemms=self.base_layer.num_gemms,
                    input_size=r,
                    output_size=out_features,
                    bias=lora_bias,
                    **kwargs,
                )
            else:
                lora_a = TELinear(
                    input_size=self.in_features,
                    output_size=r,
                    bias=lora_bias,
                    parallel_mode=None,
                    skip_weight_param_allocation=False,
                    **kwargs)
                lora_b = TEColumnParallelLinear(
                    input_size=r,
                    output_size=out_features,
                    bias=lora_bias,
                    gather_output=False,
                    **kwargs,
                )
                lora_b.parallel_mode = self.base_layer.parallel_mode  # fix moe_shared_expert_overlap
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/shared_experts.py#L93
        for lora in [lora_a, lora_b]:
            if isinstance(lora, (TERowParallelLinear, TEColumnParallelLinear)) and lora.parallel_mode is None:
                lora.ub_overlap_rs_fprop = False
                lora.ub_overlap_ag_dgrad = False
                lora.ub_overlap_ag_fprop = False
                lora.ub_overlap_rs_dgrad = False
        lora_a.sequence_parallel = False
        lora_b.sequence_parallel = False
        self.lora_A[adapter_name] = lora_a
        self.lora_B[adapter_name] = lora_b
        if hasattr(self, 'lora_bias'):
            self.lora_bias[adapter_name] = lora_bias
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / (r**0.5)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            lora_a = self.lora_A[adapter_name]
            lora_b = self.lora_B[adapter_name]
            if isinstance(lora_a, TEGroupedLinear):
                weights_a = [getattr(lora_a, f'weight{i}') for i in range(lora_a.num_gemms)]
            else:
                weights_a = [lora_a.weight]
            if isinstance(lora_b, TEGroupedLinear):
                weights_b = [getattr(lora_b, f'weight{i}') for i in range(lora_b.num_gemms)]
            else:
                weights_b = [lora_b.weight]
            for weight_a in weights_a:
                if init_lora_weights is True:
                    # initialize A the same way as the default for nn.Linear and B to zero
                    # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                    nn.init.kaiming_uniform_(weight_a, a=math.sqrt(5))
                elif init_lora_weights.lower() == 'gaussian':
                    nn.init.normal_(weight_a, std=1 / self.r[adapter_name])
                else:
                    raise ValueError(f'Unknown initialization {init_lora_weights=}')
            for weight_b in weights_b:
                nn.init.zeros_(weight_b)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    @contextmanager
    def _patch_router_gating(self):
        origin_gating = self.base_layer.__class__.gating

        def gating(_self, x):
            result = origin_gating(_self, x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(result.dtype)

                lora_result = F.linear(dropout(x), lora_A.weight.to(result.dtype))
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = F.linear(lora_result, lora_B.weight.to(result.dtype))
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_result * scaling

                result = result + lora_result
            return result

        self.base_layer.__class__.gating = gating
        try:
            yield
        finally:
            self.base_layer.__class__.gating = origin_gating

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        previous_dtype = x.dtype
        if self.disable_adapters and self.merged:
            self.unmerge()

        if isinstance(self.base_layer, TELayerNormColumnParallelLinear):
            if self.disable_adapters or self.merged:
                self.base_layer.return_layernorm_output = False
                result, bias = self.base_layer(x, *args, **kwargs)
            else:
                self.base_layer.return_layernorm_output = True
                (result, x), bias = self.base_layer(x, *args, **kwargs)
        elif isinstance(self.base_layer, (TELinear, TEGroupedLinear)):
            result, bias = self.base_layer(x, *args, **kwargs)
        elif isinstance(self.base_layer, TopKRouter):
            with self._patch_router_gating():
                result, bias = self.base_layer(x, *args, **kwargs)
        else:
            raise ValueError(f'Unsupported base layer type: {type(self.base_layer)}')
        if not isinstance(self.base_layer, TopKRouter) and not self.disable_adapters and not self.merged:
            if self.sequence_parallel and self.base_layer.parallel_mode == 'column':
                x = gather_from_sequence_parallel_region(x)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                dtype = lora_A.weight0.dtype if isinstance(lora_A, TEGroupedLinear) else lora_A.weight.dtype
                x = x.to(dtype)

                lora_result = lora_A(dropout(x), *args, **kwargs) if isinstance(lora_A, TEGroupedLinear) else lora_A(
                    dropout(x))
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_B(lora_result, *args, **kwargs) if isinstance(
                    lora_B, TEGroupedLinear) else lora_B(lora_result)
                if isinstance(lora_result, tuple):
                    lora_result = lora_result[0]
                lora_result = lora_result * scaling
                if self.sequence_parallel and self.base_layer.parallel_mode == 'row':
                    lora_result = scatter_to_sequence_parallel_region(lora_result)
                result = result + lora_result

        result = result.to(previous_dtype)
        return result, bias

    def sharded_state_dict(
            self,
            prefix: str = '',
            sharded_offsets: Tuple[Tuple[int, int, int]] = (),
            metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        sharded_state_dict = tuners_sharded_state_dict(self, prefix, sharded_offsets, metadata)
        if prefix.endswith('linear_fc1.'):
            if isinstance(self.base_layer, TEGroupedLinear) and self.config.gated_linear_unit:
                num_global_experts = (parallel_state.get_expert_model_parallel_world_size() * self.base_layer.num_gemms)
                local_expert_indices_offset = (
                    parallel_state.get_expert_model_parallel_rank() * self.base_layer.num_gemms)
                ep_axis = len(sharded_offsets)
                for i in range(self.base_layer.num_gemms):
                    new_sharded_offsets = (
                        *sharded_offsets,
                        (ep_axis, local_expert_indices_offset + i, num_global_experts),
                    )
                    for k in (f'{prefix}base_layer.weight{i}', f'{prefix}base_layer.bias{i}'):
                        if k in sharded_state_dict:
                            sharded_state_dict[k] = apply_swiglu_sharded_factory(sharded_state_dict[k],
                                                                                 new_sharded_offsets)
            else:
                for k, v in sharded_state_dict.items():
                    if k in [f'{prefix}base_layer.weight', f'{prefix}base_layer.bias']:
                        sharded_state_dict[k] = apply_swiglu_sharded_factory(sharded_state_dict[k], sharded_offsets)
        return sharded_state_dict

    def get_delta_weights(self, adapter) -> List[torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]
        if self.is_grouped:
            weight_A = [getattr(lora_A, f'weight{i}') for i in range(lora_A.num_gemms)]
            weight_B = [getattr(lora_B, f'weight{i}') for i in range(lora_B.num_gemms)]
        else:
            weight_A = [self.lora_A[adapter].weight]
            weight_B = [self.lora_B[adapter].weight]
        output_tensor = []
        assert len(weight_A) == len(weight_B)
        for i in range(len(weight_B)):
            output_tensor.append(transpose(weight_B[i] @ weight_A[i], self.fan_in_fan_out) * self.scaling[adapter])

        return output_tensor

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        base_layer = self.get_base_layer()
        origin_device = base_layer.weight0.device if self.is_grouped else base_layer.weight.device
        if origin_device.type == 'cpu':
            self.to(device=get_current_device())
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                if self.is_grouped:
                    orig_weights = [getattr(base_layer, f'weight{i}') for i in range(base_layer.num_gemms)]
                else:
                    orig_weights = [base_layer.weight]
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = [weight.data.clone() for weight in orig_weights]
                    delta_weights = self.get_delta_weights(active_adapter)
                    for orig_weight, delta_weight in zip(orig_weights, delta_weights):
                        orig_weight += delta_weight
                    if not all(torch.isfinite(orig_weights[i]).all() for i in range(len(orig_weights))):
                        raise ValueError(
                            f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    if self.is_grouped:
                        for i in range(base_layer.num_gemms):
                            weight = getattr(base_layer, f'weight{i}')
                            weight.data = orig_weights[i]
                    else:
                        base_layer.weight.data = orig_weights[0]
                else:
                    delta_weights = self.get_delta_weights(active_adapter)
                    for orig_weight, delta_weight in zip(orig_weights, delta_weights):
                        orig_weight.data += delta_weight
                self.merged_adapters.append(active_adapter)
        if origin_device.type == 'cpu':
            self.to(device=origin_device)


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

    linear_cls = (TELayerNormColumnParallelLinear, TELinear, TEGroupedLinear, TopKRouter)
    if isinstance(target_base_layer, linear_cls):
        new_module = LoraParallelLinear(base_layer=target, adapter_name=adapter_name, **kwargs)

    return new_module


model.dispatch_megatron = dispatch_megatron

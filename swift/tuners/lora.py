# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
import re
from dataclasses import dataclass, field
from types import MethodType
from typing import List, Union

import torch
from peft.tuners.lora import LoraLayer
from peft.utils import get_auto_gptq_quant_linear, get_quantization_config

from swift.utils.torch_utils import find_sub_module
from .lora_layers import * # noqa
from .utils import SwiftAdapter, SwiftConfig, SwiftOutput

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
        use_qa_lora(bool): Use
            QA-LoRA:[Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717)
            instead of LoRA. QA-LoRA only supports AutoGPTQ quantized models.
    """

    r: int = field(default=6, metadata={'help': 'The rank of the LoRA module'})
    target_modules: Union[str, List[str]] = field(
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


class LoRA(SwiftAdapter):

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
                        adapter_name,
                        sub_module,
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
                        adapter_name,
                        sub_module,
                        bias=hasattr(sub_module, 'bias')
                        and sub_module.bias is not None,
                        **four_bit_kwargs)
                elif AutoGPTQQuantLinear is not None and isinstance(
                        sub_module, AutoGPTQQuantLinear):
                    lora_module = QuantLinear(
                        adapter_name,
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
                            adapter_name,
                            sub_module.in_features,
                            sub_module.out_features,
                            bias=hasattr(sub_module, 'bias')
                            and sub_module.bias is not None,
                            **kwargs)
                elif isinstance(sub_module, torch.nn.Embedding):
                    lora_module = Embedding(
                        adapter_name,
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
                        adapter_name,
                        sub_module.in_channels,
                        sub_module.out_channels,
                        kernel_size=sub_module.kernel_size,
                        stride=sub_module.stride,
                        padding=sub_module.padding,
                        dilation=sub_module.dilation,
                        groups=sub_module.groups,
                        **kwargs)
                elif isinstance(sub_module, LoraLayer):
                    sub_module.update_layer(
                        adapter_name,
                        kwargs['r'],
                        kwargs['lora_alpha'],
                        kwargs['lora_dropout'],
                        True,
                    )

                if lora_module is not None:
                    lora_module.weight = sub_module.weight
                    if getattr(sub_module, 'bias', None) is not None:
                        lora_module.bias = sub_module.bias
                    if getattr(sub_module, 'state', None) is not None:
                        lora_module.state = sub_module.state
                    lora_module.to(sub_module.weight.device)
                    setattr(sub_module, f'loramodule_{adapter_name}',
                            lora_module)
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

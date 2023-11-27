# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
import re
from dataclasses import dataclass, field
from typing import List, Union

import torch
from peft.tuners.lora import LoraLayer
from peft.utils import get_quantization_config

from swift.utils.torch_utils import find_sub_module
from .lora_layers import *  # noqa
from .utils import SwiftAdapter, SwiftOutput, SwiftConfig
from swift import LoraConfig

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
            if isinstance(sub_module, LoraLayer):
                sub_module.set_activation(adapter_name, activate)

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
                parent = model.get_submodule(".".join(module_key.split(".")[:-1]))
                sub_module_name = module_key.split(".")[-1]

                lora_module = None
                if isinstance(sub_module, LoraLayer) and isinstance(sub_module, torch.nn.Conv2d):
                    sub_module.update_layer_conv2d(
                        adapter_name,
                        kwargs['r'],
                        kwargs['lora_alpha'],
                        kwargs['lora_dropout'],
                        True,
                    )
                elif isinstance(sub_module, LoraLayer) and isinstance(sub_module, torch.nn.Embedding):
                    sub_module.update_layer_embedding(
                        adapter_name,
                        kwargs['r'],
                        kwargs['lora_alpha'],
                        kwargs['lora_dropout'],
                        True,
                    )

                elif isinstance(sub_module, LoraLayer):
                    sub_module.update_layer(
                        adapter_name,
                        kwargs['r'],
                        kwargs['lora_alpha'],
                        kwargs['lora_dropout'],
                        True,
                    )
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

                    setattr(parent, child_name, new_module)
                    # It's not necessary to set requires_grad here, as that is handled by
                    # _mark_only_adapters_as_trainable

                    # child layer wraps the original module, unpack it
                    if hasattr(child, "base_layer"):
                        child = child.base_layer

                    if not hasattr(new_module, "base_layer"):
                        new_module.weight = child.weight
                        if hasattr(child, "bias"):
                            new_module.bias = child.bias

                    if getattr(child, "state", None) is not None:
                        if hasattr(new_module, "base_layer"):
                            new_module.base_layer.state = child.state
                        else:
                            new_module.state = child.state
                        new_module.to(child.weight.device)

                    # dispatch to correct device
                    for name, module in new_module.named_modules():
                        if (self.prefix in name) or ("ranknum" in name):
                            weight = child.qweight if hasattr(child, "qweight") else child.weight
                            module.to(weight.device)

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
        LoraModel(model, None, None).merge_and_unload(adapter_name)

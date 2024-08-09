# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
import importlib
import math
import re
import warnings
from itertools import chain
from typing import Any, Dict, List, Optional

import importlib_metadata
import packaging
import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import Conv2d as _Conv2d
from peft.tuners.lora import Embedding as _Embedding
from peft.tuners.lora import Linear as _Linear
from peft.tuners.lora import LoraLayer
from peft.tuners.lora import LoraModel as _LoraModel
from peft.tuners.lora.tp_layer import LoraParallelLinear as _LoraParallelLinear
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import _get_submodules, get_auto_gptq_quant_linear, get_quantization_config
from transformers import Conv1D

from swift import LoraConfig, get_logger
from .utils import ActivationMixin, ModulesToSaveWrapper, SwiftAdapter

logger = get_logger()
dispatchers = []


def is_auto_awq_available():
    return importlib.util.find_spec('awq') is not None


def is_aqlm_available():
    return importlib.util.find_spec('aqlm') is not None


def is_eetq_available():
    return importlib.util.find_spec('eetq') is not None


def is_hqq_available():
    return importlib.util.find_spec('hqq') is not None


def is_auto_gptq_available():
    try:
        return peft.import_utils._is_auto_gptq_available()
    except ImportError as e:
        logger.warn(e)
        return False


peft.import_utils._is_auto_gptq_available = peft.import_utils.is_auto_gptq_available
peft.import_utils.is_auto_gptq_available = is_auto_gptq_available


class LoRAActivationMixin(ActivationMixin):

    @property
    def active_adapters(self):
        return self.get_activated_adapters()

    @property
    def active_adapter(self) -> str:
        return self.get_activated_adapters()

    def set_adapter(self, adapter_names, offload=None):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Deactivate grads on the inactive adapter and activate grads on the active adapter
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key in adapter_names:
                    self.set_activation(key, True)
                    layer.requires_grad_(True)
                    SwiftAdapter.save_memory(layer, key, self.module_key, True)
                else:
                    self.set_activation(key, False)
                    layer.requires_grad_(False)
                    SwiftAdapter.save_memory(layer, key, self.module_key, False, offload=offload)

    def save_memory(self, adapter_name, activate, offload=None):
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, layer in module_dict.items():
                if key == adapter_name:
                    if activate:
                        SwiftAdapter.save_memory(layer, layer_name + '.' + key, self.module_key, True)
                    else:
                        SwiftAdapter.save_memory(layer, layer_name + '.' + key, self.module_key, False, offload=offload)

    def merge(self, *args, **kwargs):
        if not self.unique_thread:
            raise AssertionError('Merge is unsupported in multiple thread, '
                                 'please set `USE_UNIQUE_THREAD=1` in env variable to merge LoRA.')
        return super().merge(*args, **kwargs)


if is_bnb_available():
    import bitsandbytes as bnb
    from peft.tuners.lora.bnb import Linear8bitLt as _Linear8bitLt

    class Linear8bitLt(LoRAActivationMixin, _Linear8bitLt):

        def __init__(
            self,
            *args,
            module_key: str,
            **kwargs,
        ):
            super(Linear8bitLt, self).__init__(module_key)
            self.set_activation(args[1], True)
            super(ActivationMixin, self).__init__(*args, **kwargs)

    def dispatch_bnb_8bit(target: torch.nn.Module, adapter_name: str, module_key: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_8bit = kwargs.get('loaded_in_8bit', False)
        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update({
                'has_fp16_weights': target.state.has_fp16_weights,
                'memory_efficient_backward': target.state.memory_efficient_backward,
                'threshold': target.state.threshold,
                'index': target.index,
            })
            new_module = Linear8bitLt(target, adapter_name, module_key=module_key, **eightbit_kwargs)

        return new_module

    dispatchers.append(dispatch_bnb_8bit)

if is_bnb_4bit_available():
    from peft.tuners.lora.bnb import Linear4bit as _Linear4bit

    class Linear4bit(LoRAActivationMixin, _Linear4bit):

        def __init__(
            self,
            *args,
            module_key: str,
            **kwargs,
        ):
            super(Linear4bit, self).__init__(module_key)
            self.set_activation(args[1], True)
            super(ActivationMixin, self).__init__(*args, **kwargs)

    def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, module_key: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_4bit = kwargs.get('loaded_in_4bit', False)
        if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update({
                'compute_dtype': target_base_layer.compute_dtype,
                'compress_statistics': target_base_layer.weight.compress_statistics,
                'quant_type': target_base_layer.weight.quant_type,
            })
            new_module = Linear4bit(target, adapter_name, module_key=module_key, **fourbit_kwargs)

        return new_module

    dispatchers.append(dispatch_bnb_4bit)

if is_aqlm_available():
    from peft.tuners.lora.aqlm import AqlmLoraLinear as _AqlmLoraLinear
    from aqlm import QuantizedLinear

    class AqlmLoraLinear(LoRAActivationMixin, _AqlmLoraLinear):

        def __init__(
            self,
            *args,
            module_key: str,
            **kwargs,
        ):
            super(AqlmLoraLinear, self).__init__(module_key)
            self.set_activation(args[1], True)
            super(ActivationMixin, self).__init__(*args, **kwargs)

    def dispatch_aqlm(
        target: torch.nn.Module,
        adapter_name: str,
        **kwargs: Any,
    ) -> Optional[torch.nn.Module]:
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if is_aqlm_available() and isinstance(target_base_layer, QuantizedLinear):
            new_module = AqlmLoraLinear(target, adapter_name, **kwargs)
            target.qweight = target_base_layer.codes

        return new_module

    dispatchers.append(dispatch_aqlm)

if is_auto_awq_available():
    from peft.tuners.lora.awq import AwqLoraLinear as _AwqLoraLinear
    from awq.modules.linear import WQLinear_GEMM

    class AwqLoraLinear(LoRAActivationMixin, _AwqLoraLinear):

        def __init__(
            self,
            *args,
            module_key: str,
            **kwargs,
        ):
            super(AwqLoraLinear, self).__init__(module_key)
            self.set_activation(args[1], True)
            super(ActivationMixin, self).__init__(*args, **kwargs)

    def dispatch_awq(
        target: torch.nn.Module,
        adapter_name: str,
        module_key: str,
        **kwargs: Any,
    ) -> Optional[torch.nn.Module]:
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, WQLinear_GEMM):
            # Raise the error only at the dispatch level
            AUTOAWQ_MINIMUM_VERSION = packaging.version.parse('0.2.0')
            version_autoawq = packaging.version.parse(importlib_metadata.version('autoawq'))

            if AUTOAWQ_MINIMUM_VERSION > version_autoawq:
                raise ImportError(f'Found an incompatible version of auto-awq. Found version {version_autoawq}, '
                                  f'but only versions above {AUTOAWQ_MINIMUM_VERSION} are supported for PEFT.')

            new_module = AwqLoraLinear(target, adapter_name, module_key=module_key, **kwargs)
            target.qweight = target_base_layer.qweight

        return new_module

    dispatchers.append(dispatch_awq)

if is_auto_gptq_available():
    from peft.tuners.lora import QuantLinear as _QuantLinear

    class QuantLinear(LoRAActivationMixin, _QuantLinear):

        def __init__(
            self,
            base_layer,
            adapter_name: str,
            module_key: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            use_qa_lora=False,
            group_size=None,
            **kwargs,
        ):
            super(QuantLinear, self).__init__(module_key)
            self.set_activation(adapter_name, True)
            nn.Module.__init__(self)
            self.group_size = group_size
            self.use_qa_lora = use_qa_lora
            if self.use_qa_lora:
                assert self.group_size is not None, 'To use qa_lora you need to pass in the `group_size` param.'
                self.qa_pool = torch.nn.AvgPool1d(self.group_size)  # using pooling layer to conduct sum operation

            LoraLayer.__init__(self, base_layer)
            if use_dora:
                raise ValueError(f'{_QuantLinear.__name__} does not support DoRA yet, please set it to False')
            if self.use_qa_lora:
                self.in_features = self.in_features // self.group_size
            # self.base_layer and self.quant_linear_module are the same;
            # we need the former for consistency and the latter
            # for backwards compatibility
            self.quant_linear_module = base_layer
            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
            )

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

    def dispatch_gptq(
        target: torch.nn.Module,
        adapter_name: str,
        module_key: str,
        **kwargs: Any,
    ) -> Optional[torch.nn.Module]:
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        gptq_quantization_config = kwargs.get('gptq_quantization_config', None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        if AutoGPTQQuantLinear is not None and isinstance(target_base_layer, AutoGPTQQuantLinear):
            new_module = QuantLinear(target, adapter_name, module_key=module_key, **kwargs)
            target.qweight = target_base_layer.qweight

        return new_module

    dispatchers.append(dispatch_gptq)

if is_eetq_available():
    from peft.tuners.lora.eetq import EetqLoraLinear as _EetqLoraLinear
    from eetq import EetqLinear

    class EetqLoraLinear(LoRAActivationMixin, _EetqLoraLinear):

        def __init__(
            self,
            *args,
            module_key: str,
            **kwargs,
        ):
            super(EetqLoraLinear, self).__init__(module_key)
            self.set_activation(args[1], True)
            super(ActivationMixin, self).__init__(*args, **kwargs)

    def dispatch_eetq(
        target: torch.nn.Module,
        adapter_name: str,
        **kwargs: Any,
    ) -> Optional[torch.nn.Module]:
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if is_eetq_available() and isinstance(target_base_layer, EetqLinear):
            new_module = EetqLoraLinear(target, adapter_name, **kwargs)
            target.weight = target_base_layer.weight

            if hasattr(target, 'bias'):
                target.bias = target_base_layer.bias

        return new_module

    dispatchers.append(dispatch_eetq)

if is_hqq_available():
    from peft.tuners.lora.hqq import HqqLoraLinear as _HqqLoraLinear
    from hqq.core.quantize import HQQLinear

    class HqqLoraLinear(LoRAActivationMixin, _HqqLoraLinear):

        def __init__(
            self,
            *args,
            module_key: str,
            **kwargs,
        ):
            super(HqqLoraLinear, self).__init__(module_key)
            self.set_activation(args[1], True)
            super(ActivationMixin, self).__init__(*args, **kwargs)

    def dispatch_hqq(target: torch.nn.Module, adapter_name: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if is_hqq_available() and isinstance(target_base_layer, HQQLinear):
            new_module = HqqLoraLinear(target_base_layer, adapter_name, **kwargs)

        return new_module

    dispatchers.append(dispatch_hqq)


def dispatch_megatron(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config,
    module_key,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if lora_config.megatron_config:
        megatron_core = importlib.import_module(lora_config.megatron_core)
    else:
        megatron_core = None

    if megatron_core and isinstance(
            target_base_layer,
        (megatron_core.tensor_parallel.ColumnParallelLinear, megatron_core.tensor_parallel.RowParallelLinear)):  # noqa
        megatron_kwargs = kwargs.copy()
        megatron_config = lora_config.megatron_config
        if isinstance(megatron_config, dict):
            transformer_config_class = megatron_core.transformer.transformer_config.TransformerConfig
            megatron_config = transformer_config_class(**lora_config.megatron_config)
        megatron_kwargs['megatron_config'] = megatron_config
        if megatron_kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to True but the target module is `ColumnParallelLinear` '
                          'or `RowParallelLinear`. '
                          'Setting fan_in_fan_out to False.')
            megatron_kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = False
        new_module = LoraParallelLinear(
            base_layer=target,
            adapter_name=adapter_name,
            module_key=module_key,
            backend=megatron_core.tensor_parallel,
            **megatron_kwargs)

    return new_module


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    module_key: str,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop('fan_in_fan_out', None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, module_key=module_key, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, module_key=module_key, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if target_base_layer.__class__.__name__ == 'NonDynamicallyQuantizableLinear':
            # Fix issue: https://github.com/modelscope/swift/issues/342
            return None
        if kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. '
                          'Setting fan_in_fan_out to False.')
            kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, module_key=module_key, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs['fan_in_fan_out']:
            warnings.warn('fan_in_fan_out is set to False but the target module is `Conv1D`. '
                          'Setting fan_in_fan_out to True.')
            kwargs['fan_in_fan_out'] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, module_key=module_key, **kwargs)

    return new_module


dispatchers.append(dispatch_megatron)
dispatchers.append(dispatch_default)


class Embedding(LoRAActivationMixin, _Embedding):

    def __init__(
        self,
        *args,
        module_key: str,
        **kwargs,
    ) -> None:
        super(Embedding, self).__init__(module_key)
        self.set_activation(args[1], True)
        super(ActivationMixin, self).__init__(*args, **kwargs)


class Linear(LoRAActivationMixin, _Linear):

    def __init__(self, *args, module_key: str, **kwargs):
        super(Linear, self).__init__(module_key)
        self.set_activation(args[1], True)
        super(ActivationMixin, self).__init__(*args, **kwargs)


class Conv2d(LoRAActivationMixin, _Conv2d):

    def __init__(self, *args, module_key: str, **kwargs):
        super(Conv2d, self).__init__(module_key)
        self.set_activation(args[1], True)
        super(ActivationMixin, self).__init__(*args, **kwargs)


class LoraParallelLinear(LoRAActivationMixin, _LoraParallelLinear):

    def __init__(self, *args, module_key: str, **kwargs):
        super(LoraParallelLinear, self).__init__(module_key)
        self.set_activation(args[1], True)
        super(ActivationMixin, self).__init__(*args, **kwargs)


class LoraModel(_LoraModel):

    prefix: str = 'lora_'

    def __init__(self, model, config, adapter_name):
        if config is not None:
            super().__init__(model, config, adapter_name)
        else:
            nn.Module.__init__(self)
            self.model = model

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == 'none':
                continue

            if bias == 'all':
                for n, p in model.named_parameters():
                    if 'bias' in n:
                        p.requires_grad = True
            elif bias == 'lora_only':
                for m in model.modules():
                    if isinstance(m, LoraLayer) and hasattr(m, 'bias') and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f'Requested bias: {bias}, is not implemented.')

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        r"""
        Override code:
        1. ModulesToSaveWrapper construction method: add module_key=key argument to offload to cpu
        """
        peft_config = self.peft_config[adapter_name]
        # Note: If possible, all checks should be performed *at the start of this method*.
        # This way, we can raise early if something goes wrong, without leaving the model
        # in a bad (half-initialized) state.
        self._check_new_adapter_config(peft_config)

        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]

        _check_for_modules_to_save = getattr(peft_config, 'modules_to_save', None) is not None
        _has_modules_to_save = False

        model_config = getattr(model, 'config', {'model_type': 'custom'})
        if hasattr(model_config, 'to_dict'):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(peft_config, model_config)

        from peft.tuners.tuners_utils import _maybe_include_all_linear_layers
        try:
            from peft.utils.constants import DUMMY_TARGET_MODULES
        except ImportError:  # compat with peft==0.11.*
            DUMMY_TARGET_MODULES = 'dummy-target-modules'
        if getattr(peft_config, 'target_modules', None) == DUMMY_TARGET_MODULES:
            # dummy adapter, we allow not matching any module
            key_list = []
            is_target_modules_in_base_model = True
        # update peft_config.target_modules if required
        peft_config = _maybe_include_all_linear_layers(peft_config, model)
        self._prepare_model(peft_config, model)

        for key in key_list:
            if '_part_' in key:
                # Avoid lora conflict with part tuner
                continue
            # Check for modules_to_save in case
            if _check_for_modules_to_save and any(
                    key.endswith(f'{module_to_save}') for module_to_save in peft_config.modules_to_save):
                # Optionally set the modules to save
                parent, target, target_name = _get_submodules(model, key)

                if not isinstance(target, ModulesToSaveWrapper):
                    new_module = ModulesToSaveWrapper(target, adapter_name=adapter_name, module_key=key)
                    setattr(parent, target_name, new_module)
                else:
                    target.update(adapter_name)

                _has_modules_to_save = True
                continue

            if not self._check_target_module_exists(peft_config, key):
                continue

            self.targeted_module_names.append(key)
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)

        # Handle X-LoRA case.
        if not is_target_modules_in_base_model and hasattr(peft_config, 'target_modules'):
            raise ValueError(f'Target modules {peft_config.target_modules} not found in the base model. '
                             f'Please check the target modules and try again.')

        self._mark_only_adapters_as_trainable(self.model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in self.model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False

        if _has_modules_to_save:
            if not hasattr(model, 'modules_to_save'):
                model.modules_to_save = set(peft_config.modules_to_save)
            else:
                model.modules_to_save.update(set(peft_config.modules_to_save))

    def _convert_dtype(self, target: nn.Module, lora_dtype: str):
        if lora_dtype == 'fp32':
            torch_dtype = torch.float32
        elif lora_dtype == 'fp16':
            torch_dtype = torch.float16
        elif lora_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None

        if torch_dtype is not None:
            if hasattr(target, 'lora_A'):
                target.lora_A.to(torch_dtype)
                target.lora_B.to(torch_dtype)
            if hasattr(target, 'lora_embedding_A'):
                target.lora_embedding_A.to(torch_dtype)
                target.lora_embedding_B.to(torch_dtype)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        """
        Override code:
        1. Import bnb from upper code
        2. Support dtype converting
        3. Support skipping NonDynamicallyQuantizableLinear
        4. Add current_key argument to _create_new_module
        5. Use Class type defined here
        6. Allow new_module being None
        """
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf'.*\.{key}$', current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            'r': r,
            'lora_alpha': alpha,
            'lora_dropout': lora_config.lora_dropout,
            'fan_in_fan_out': lora_config.fan_in_fan_out,
            'init_lora_weights': lora_config.init_lora_weights,
            'use_rslora': lora_config.use_rslora,
            'use_dora': lora_config.use_dora,
            'loaded_in_8bit': getattr(self.model, 'is_loaded_in_8bit', False),
            'loaded_in_4bit': getattr(self.model, 'is_loaded_in_4bit', False),
        }
        # compat with peft==0.11.*
        if hasattr(lora_config, 'runtime_config'):
            kwargs['ephemeral_gpu_offload'] = lora_config.runtime_config.ephemeral_gpu_offload

        quant_methods = ['gptq', 'aqlm', 'awq']
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f'{quant_method}_quantization_config'] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            if target.__class__.__name__ == 'NonDynamicallyQuantizableLinear':
                # Fix issue: https://github.com/modelscope/swift/issues/342
                return
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
            self._convert_dtype(target, lora_config.lora_dtype)
            ActivationMixin.mark_all_sub_modules_as_plugin(target)
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, current_key=current_key, **kwargs)
            if new_module is not None:
                ActivationMixin.mark_all_sub_modules_as_plugin(new_module)
                if adapter_name not in self.active_adapters:
                    # adding an additional adapter: it is not automatically trainable
                    new_module.requires_grad_(False)
                self._replace_module(parent, target_name, new_module, target)
                self._convert_dtype(new_module, lora_config.lora_dtype)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, 'base_layer'):
            child = child.base_layer

        if not hasattr(new_module, 'base_layer'):
            if hasattr(new_module, 'W_q'):  # HQQ
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight
            if hasattr(child, 'bias'):
                new_module.bias = child.bias

        if getattr(child, 'state', None) is not None:
            if hasattr(new_module, 'base_layer'):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ('ranknum' in name):
                weight = (
                    child.qweight if hasattr(child, 'qweight') else child.W_q if hasattr(child, 'W_q') else
                    child.weight if hasattr(child, 'weight') else next(child.parameters()))
                module.to(weight.device)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        """
        Override code:
        1. Support current_key argument
        2. Support MergedLinear
        3. Support skipping NonDynamicallyQuantizableLinear(Move to dispatcher)
        4. Use Class type defined here(Move to dispatcher)
        5. return None instead of raising error when target type not found
        """
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        current_key = kwargs.pop('current_key')
        new_module = None
        if lora_config.use_qa_lora:
            kwargs['use_qa_lora'] = True
            kwargs['group_size'] = lora_config.group_size
        if lora_config.use_merged_linear:
            bias = kwargs.pop('bias', False)
            new_module = MergedLinear(
                adapter_name, current_key, target, bias=bias, enable_lora=lora_config.enable_lora, **kwargs)
        else:
            for dispatcher in dispatchers:
                new_module = dispatcher(target, adapter_name, lora_config=lora_config, module_key=current_key, **kwargs)
                if new_module is not None:  # first match wins
                    break

        if new_module is None:
            # no module could be matched
            logger.debug(
                f'Target module {target} is not supported. Currently, only the following modules are supported: '
                '`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`.')
            new_module = None

        return new_module


class LoRALayer(ActivationMixin):

    def __init__(
        self,
        adapter_name: str,
        module_key: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        super().__init__(module_key)
        self.adapter_name = adapter_name
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
                 adapter_name: str,
                 module_key: str,
                 base_layer: nn.Linear,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.,
                 enable_lora: List[bool] = [False],
                 fan_in_fan_out: bool = False,
                 merge_weights: bool = True,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 **kwargs):
        nn.Linear.__init__(self, base_layer.in_features, base_layer.out_features, bias=bias, device=device, dtype=dtype)
        LoRALayer.__init__(
            self,
            adapter_name,
            module_key,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights)
        assert base_layer.out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.base_layer = base_layer
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), base_layer.in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((base_layer.out_features // len(enable_lora) * sum(enable_lora),
                                       r)))  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((base_layer.out_features, ),
                                                  dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        self.weight = self.base_layer.weight
        if getattr(self.base_layer, 'bias', None) is not None:
            self.bias = self.base_layer.bias
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

        delta_w = F.conv1d(self.lora_A.unsqueeze(0), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
        return T(self.zero_pad(delta_w))

    def merge(self, **kwargs):
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                self.weight.data += self.merge_AB() * self.scaling

    def unmerge(self, **kwargs):
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                self.weight.data -= self.merge_AB() * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged or not self.is_activated(self.adapter_name):
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                x_dtype = x.dtype
                x = x.to(self.lora_A.dtype)
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
                result = result.to(x_dtype)
            return result


def mark_lora_as_trainable(model: nn.Module, adapter_name: str, bias: str = 'none') -> None:
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for n, m in model.named_modules():
            if 'lora_' in n and f'.{adapter_name}' in n and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(state_dict, adapter_name: str, bias: str = 'none') -> Dict[str, torch.Tensor]:
    if bias == 'none':
        to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
    elif bias == 'all':
        to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in state_dict:
            if 'lora_' in k:
                to_return[k] = state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    return {k: v for k, v in to_return.items() if (('lora_' in k and f'.{adapter_name}' in k) or ('bias' in k))}

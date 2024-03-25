# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import os.path
from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
import torch.nn.functional as F
from peft.utils.other import transpose

from swift import get_logger

logger = get_logger()
dispatchers = []
import torch.nn
from peft import (AdaLoraConfig, IA3Config, LoftQConfig, LoHaConfig,
                  LoKrConfig, OFTConfig, PeftConfig, PeftModel,
                  PeftModelForCausalLM, PeftModelForSeq2SeqLM,
                  PeftModelForSequenceClassification,
                  PeftModelForTokenClassification, PrefixTuningConfig,
                  PromptEncoderConfig, PromptLearningConfig,
                  PromptTuningConfig, get_peft_config, get_peft_model,
                  get_peft_model_state_dict, LoraModel)

from swift.hub.snapshot_download import snapshot_download


@dataclass
class LoraConfig(peft.LoraConfig):
    lora_dtype: str = field(
        default=None,
        metadata={
            'help':
            'The lora dtype, default None means following the original layer\'s dtype'
        })

    lr_ratio: float = field(
        default=2.0**4,
        metadata={'help': 'The lora learning_rate ratio of lora_A to lora_B'})


def _apply_dora(self, x, lora_A, lora_B, scaling, active_adapter):
    """
    From LoraLayer._apply_dora, to support `weight.to(x.dtype)`
    """
    lora_weight = lora_B.weight @ lora_A.weight
    magnitude = self.lora_magnitude_vector[active_adapter]
    weight = self.get_base_layer().weight
    weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
    weight_norm = weight_norm.detach()
    mag_norm_scale = (magnitude / weight_norm).view(1, -1)
    result_dora = (mag_norm_scale - 1) * (F.linear(
        x, transpose(weight.to(x.dtype), self.fan_in_fan_out)
    )) + mag_norm_scale * lora_B(lora_A(x)) * scaling
    return result_dora


def _create_and_replace_hook(
    self, *args, **kwargs
):
    target = None
    if 'target' in kwargs:
        target = kwargs['target']
    else:
        for arg in args:
            if isinstance(arg, torch.nn.Module):
                target = arg
                break

    if target and target.__class__.__name__ == 'NonDynamicallyQuantizableLinear':
        return

    return self._create_and_replace_origin(*args, **kwargs)


def _convert_dtype(target: torch.nn.Module, lora_dtype: str):
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


def create_optimizer_param_groups(self, **defaults):
    all_param_names = set()
    param_groups = []
    for output in self.adapters.values():
        if output.optimizer_group_callback:
            param_names, param_group = output.optimizer_group_callback(
                self.model, **defaults)
            if param_names and all_param_names & param_names:
                raise ValueError(
                    'Cannot set one parameter to different param groups')
            if param_names and param_group:
                all_param_names.update(param_names)
                param_groups.append(param_group)

    decay_parameters = Trainer.get_decay_parameter_names(None, self.model)
    param_groups.extend([
        {
            'params': [
                p for n, p in self.model.named_parameters()
                if (n in decay_parameters and n not in all_param_names
                    and p.requires_grad)
            ],
            'weight_decay':
            defaults['weight_decay'],
        },
        {
            'params': [
                p for n, p in self.model.named_parameters()
                if (n not in decay_parameters and n not in all_param_names
                    and p.requires_grad)
            ],
            'weight_decay':
            0.0,
        },
    ])

    return param_groups


def hot_patch_peft_module():
    from peft.tuners.lora import LoraLayer

    # Fix Lora does not support NonDynamicallyQuantizableLinear
    LoraModel._create_and_replace_origin = LoraModel._create_and_replace
    LoraModel._create_and_replace = _create_and_replace_hook

    # Fix dora dtype
    LoraLayer._apply_dora = _apply_dora

    # Support type conversion
    def init(self, model: torch.nn.Module, config: LoraConfig, adapter_name):
        self.__init_origin__(model, config, adapter_name)

        for module in model.modules():
            if isinstance(module, LoraLayer):
                _convert_dtype(module, config.lora_dtype)

    LoraModel.__init_origin__ = LoraModel.__init__
    LoraModel.__init__ = init




def get_wrapped_class(module_class):
    """Get a custom wrapper class for peft classes to download the models from the ModelScope hub

    Args:
        module_class: The actual module class

    Returns:
        The wrapper
    """

    class PeftWrapper(module_class):

        @classmethod
        def from_pretrained(cls,
                            model,
                            model_id,
                            *args,
                            revision: Optional[str] = None,
                            **kwargs):
            if not os.path.exists(model_id):
                model_id = snapshot_download(model_id, revision=revision)
            return module_class.from_pretrained(model, model_id, *args,
                                                **kwargs)

    return PeftWrapper


def wrap_module(module):
    if not hasattr(module, 'from_pretrained'):
        return module

    return get_wrapped_class(module)


PeftModel = wrap_module(PeftModel)
PeftConfig = wrap_module(PeftConfig)
PeftModelForSeq2SeqLM = wrap_module(PeftModelForSeq2SeqLM)
PeftModelForSequenceClassification = wrap_module(
    PeftModelForSequenceClassification)
PeftModelForTokenClassification = wrap_module(PeftModelForTokenClassification)
PeftModelForCausalLM = wrap_module(PeftModelForCausalLM)
PromptEncoderConfig = wrap_module(PromptEncoderConfig)
PromptTuningConfig = wrap_module(PromptTuningConfig)
PrefixTuningConfig = wrap_module(PrefixTuningConfig)
PromptLearningConfig = wrap_module(PromptLearningConfig)
LoraConfig = wrap_module(LoraConfig)
AdaLoraConfig = wrap_module(AdaLoraConfig)
IA3Config = wrap_module(IA3Config)
LoHaConfig = wrap_module(LoHaConfig)
LoKrConfig = wrap_module(LoKrConfig)
LoftQConfig = wrap_module(LoftQConfig)
OFTConfig = wrap_module(OFTConfig)
get_peft_config = get_peft_config
get_peft_model_state_dict = get_peft_model_state_dict
get_peft_model = get_peft_model

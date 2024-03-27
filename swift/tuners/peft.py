# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import os.path
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Dict, Optional

import peft
import torch
import torch.nn
import torch.nn.functional as F
from peft import (AdaLoraConfig, IA3Config, LoftQConfig, LoHaConfig,
                  LoKrConfig, LoraModel, OFTConfig, PeftConfig, PeftModel,
                  PeftModelForCausalLM, PeftModelForSeq2SeqLM,
                  PeftModelForSequenceClassification,
                  PeftModelForTokenClassification, PrefixTuningConfig,
                  PromptEncoderConfig, PromptLearningConfig,
                  PromptTuningConfig, get_peft_config, get_peft_model,
                  get_peft_model_state_dict)
from peft.tuners.lora import Embedding
from peft.utils.other import transpose
from transformers import Trainer

from swift import get_logger
from swift.hub.snapshot_download import snapshot_download

logger = get_logger()
dispatchers = []


@dataclass
class LoraConfig(peft.LoraConfig):
    lora_dtype: str = field(
        default=None,
        metadata={
            'help':
            'The lora dtype, default None means following the original layer\'s dtype'
        })

    lorap_lr_ratio: float = field(
        default=2.0**4,
        metadata={'help': 'The lr ratio of lora_B in lora+'})

    lorap_emb_lr: float = field(default=1e-6, metadata={'help': 'The lr for embedding in lora+'})


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


def _create_and_replace_hook(self, *args, **kwargs):
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


def _convert_dtype(target: torch.nn.Module, adapter_name: str, lora_dtype: str):
    if lora_dtype == 'fp32':
        torch_dtype = torch.float32
    elif lora_dtype == 'fp16':
        torch_dtype = torch.float16
    elif lora_dtype == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None

    if torch_dtype is not None:
        if hasattr(target, 'lora_A') and adapter_name in target.lora_A:
            target.lora_A[adapter_name].to(torch_dtype)
            target.lora_B[adapter_name].to(torch_dtype)
        if hasattr(target, 'lora_embedding_A') and adapter_name in target.lora_embedding_A:
            target.lora_embedding_A[adapter_name].to(torch_dtype)
            target.lora_embedding_B[adapter_name].to(torch_dtype)


def create_optimizer_param_groups(self: PeftModel, **defaults):
    if not isinstance(self.peft_config[self.active_adapter],
                  LoraConfig) or self.peft_config[self.active_adapter].lorap_lr_ratio is None:
        return None

    def get_module(name):
        parent_idx = 2 if 'lora' in name else 1
        module_names = name.split(sep='.')[:-parent_idx]
        module = reduce(getattr, module_names, self.base_model)
        return module

    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    decay_parameters = Trainer.get_decay_parameter_names(None, self.base_model)
    for name, param in self.base_model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_module(name)
        if isinstance(module, Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    lr = defaults["lr"]
    weight_decay = defaults.get("weight_decay", 0.0)

    param_groups = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": self.peft_config[self.active_adapter].lorap_emb_lr,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * self.peft_config[self.active_adapter].lorap_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * self.peft_config[self.active_adapter].lorap_lr_ratio,
        },
    ]
    return param_groups


def hot_patch_peft_module():
    from peft.tuners.lora import LoraLayer

    # Fix Lora does not support NonDynamicallyQuantizableLinear
    LoraModel._create_and_replace_origin = LoraModel._create_and_replace
    LoraModel._create_and_replace = _create_and_replace_hook

    # Fix dora dtype
    LoraLayer._apply_dora = _apply_dora

    # Support type conversion
    def init(self, model: torch.nn.Module, config: Dict[str, LoraConfig],
             adapter_name):
        self.__init_origin__(model, config, adapter_name)
        
        active_config = config[self.active_adapter] if isinstance(config, dict) else config
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                _convert_dtype(module, self.active_adapter, active_config.lora_dtype)

    LoraModel.__init_origin__ = LoraModel.__init__
    LoraModel.__init__ = init

    # Support LoRA+
    PeftModel.create_optimizer_param_groups = create_optimizer_param_groups

    # Compatible with SwiftModel
    def dummy_function(*args, **kwargs):
        logger.warn(
            f'The function {kwargs["func"]} has no effects, consider using other functions.'
        )

    PeftModel.activate_adapter = PeftModel.set_adapter
    PeftModel.deactivate_adapter = partial(
        dummy_function, func='deactivate_adapter')
    PeftModel.set_active_adapters = partial(
        dummy_function, func='set_active_adapters')


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


hot_patch_peft_module()
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

# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.

import inspect
import os
from collections import OrderedDict
from types import FunctionType, MethodType
from typing import List, Union

import torch
from torch.nn import Module
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (EvaluationStrategy, FSDPOption,
                                        HPSearchBackend, HubStrategy,
                                        IntervalStrategy, SchedulerType)

from swift.utils import get_logger

try:
    # https://github.com/huggingface/transformers/pull/25702
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    ShardedDDPOption = None

logger = get_logger()


def can_return_loss(model: Module) -> bool:
    """Check if a given model can return loss."""
    signature = inspect.signature(model.forward)
    for p in signature.parameters:
        if p == 'return_loss' and signature.parameters[p].default is True:
            return True
    return False


def find_labels(model: Module) -> List[str]:
    """Find the labels used by a given model."""
    model_name = model.__class__.__name__
    signature = inspect.signature(model.forward)
    if 'QuestionAnswering' in model_name:
        return [
            p for p in signature.parameters
            if 'label' in p or p in ('start_positions', 'end_positions')
        ]
    else:
        return [p for p in signature.parameters if 'label' in p]


def get_function(
        method_or_function: Union[MethodType, FunctionType]) -> FunctionType:
    if isinstance(method_or_function, MethodType):
        method_or_function = method_or_function.__func__
    return method_or_function


def is_instance_of_ms_model(model: Module) -> bool:
    """avoid import modelscope: circular dependency problem"""
    for m_cls in model.__class__.__mro__:
        cls_name = m_cls.__name__
        cls_module = m_cls.__module__
        if cls_name == 'Model' and cls_module.startswith('modelscope'):
            return True
    return False


def consolidate_checkpoint(resume_from_checkpoint, model_name='adapter_model'):
    """ Consolidate the sharded TorchAcc checkpoints into a single model checkpoint.
    """
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.fsdp import consolidate_sharded_state_dicts

    if model_name not in ('adapter_model', 'model'):
        logger.error('Only support PeftModel and PreTrainedModel.')
        return

    model_dir = os.path.join(resume_from_checkpoint, '0')
    is_pretrained_model = False
    if os.path.exists(os.path.join(model_dir, f'{model_name}.safetensors')):
        use_safetensors = True
    elif os.path.exists(os.path.join(model_dir, f'{model_name}.bin')):
        use_safetensors = False
    elif os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
        # PreTrainedModel use 'pytorch_model.bin' and 'model.safetensors'
        use_safetensors = False
        is_pretrained_model = True
    else:
        logger.error('Cannot find checkpoint.')

    state_dict_list = []
    if xm.is_master_ordinal(local=False) and use_safetensors:
        from safetensors.torch import load_file, save_file
        for rank in range(xm.xrt_world_size()):
            shard_dir = os.path.join(resume_from_checkpoint, f'{rank}')
            filename = os.path.join(shard_dir, f'{model_name}.safetensors')
            state_dict = load_file(filename, device='cpu')
            state_dict = OrderedDict(('_fsdp_wrapped_module.' + k, v)
                                     for k, v in state_dict.items())
            state_dict_list.append(state_dict)
        shard_metadata = torch.load(
            os.path.join(model_dir, 'shard_meta.pth'), map_location='cpu')
    elif xm.is_master_ordinal(local=False):
        for rank in range(xm.xrt_world_size()):
            shard_dir = os.path.join(resume_from_checkpoint, f'{rank}')
            if not is_pretrained_model:
                filename = os.path.join(shard_dir, f'{model_name}.bin')
            else:
                filename = os.path.join(shard_dir, 'pytorch_model.bin')
            state_dict = torch.load(filename, map_location='cpu')
            state_dict = OrderedDict(('_fsdp_wrapped_module.' + k, v)
                                     for k, v in state_dict.items())
            state_dict_list.append(state_dict)
        shard_metadata = torch.load(
            os.path.join(model_dir, 'shard_meta.pth'), map_location='cpu')

    if xm.is_master_ordinal(local=False):
        full_state_dict = consolidate_sharded_state_dicts(
            state_dict_list, shard_metadata)
        # peft will prepend "default." prefix automatically, so we remove the
        # "default." prefix to prevent the duplication of the prefix.
        full_state_dict = OrderedDict(
            (k.replace('default.', ''), v) for k, v in full_state_dict.items())
        torch.save(full_state_dict,
                   os.path.join(resume_from_checkpoint, f'{model_name}.bin'))
        if model_name == 'adapter_model':
            config_path = os.path.join(resume_from_checkpoint,
                                       'adapter_config.json')
            old_config_path = os.path.join(model_dir, 'adapter_config.json')
            os.system(f'cp {old_config_path} {config_path}')
    xm.rendezvous('ckpt_consolidation')

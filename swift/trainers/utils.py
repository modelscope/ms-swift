# Copyright (c) ModelScope Contributors. All rights reserved.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import math
import os
from contextlib import contextmanager
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from transformers import PreTrainedModel

from swift.model import ModelMeta
from swift.sequence_parallel import ChunkedCrossEntropyLoss, GatherLoss, sequence_parallel
from swift.utils import deep_getattr, get_dist_setting, get_logger

if TYPE_CHECKING:
    from .arguments import TrainingArguments

logger = get_logger()


def can_return_loss(model: Module) -> bool:
    """Check if a given model can return loss."""
    if isinstance(model, PeftModel):
        signature = inspect.signature(model.model.forward)
    else:
        signature = inspect.signature(model.forward)
    for p in signature.parameters:
        if p == 'return_loss' and signature.parameters[p].default is True:
            return True
    return False


def find_labels(model: Module) -> List[str]:
    """Find the labels used by a given model."""
    model_name = model.__class__.__name__
    if isinstance(model, PeftModel):
        signature = inspect.signature(model.model.forward)
    else:
        signature = inspect.signature(model.forward)
    if 'QuestionAnswering' in model_name:
        return [p for p in signature.parameters if 'label' in p or p in ('start_positions', 'end_positions')]
    else:
        return [p for p in signature.parameters if 'label' in p]


def get_function(method_or_function: Union[MethodType, FunctionType]) -> FunctionType:
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


def per_token_loss_func_sp(outputs, labels, enable_dft_loss=False, **kwargs) -> torch.Tensor:
    """Common loss function for sequence parallel training"""
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    device = logits.device

    batch_size = logits.shape[0]
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.flatten().to(device)
    sploss_parallel_size = int(os.environ.get('CELOSS_PARALLEL_SIZE', '0'))
    if sploss_parallel_size > 0:
        loss = ChunkedCrossEntropyLoss.apply(logits, labels, sploss_parallel_size)
    else:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs
    position_ids = sequence_parallel.real_position_ids
    if position_ids is not None:
        position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
    loss, labels = GatherLoss.apply(loss.reshape(batch_size, -1), labels.reshape(batch_size, -1), 1, position_ids)
    if position_ids is not None and position_ids.min() == -1:
        _pos_mask = position_ids >= 0
        loss = loss[_pos_mask].contiguous()

    return loss


def per_token_loss_func(outputs, labels, enable_dft_loss: bool = False, **kwargs):
    logits = outputs.logits
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)

    # Flatten the tokens
    logits = logits.view(-1, logits.shape[-1])
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs
    return loss


def _kwargs_to_args(func, args, kwargs) -> Optional[List[Any]]:
    parameters = inspect.signature(func).parameters
    args = list(args)
    parameters = list(parameters.items())[len(args):]
    for key, param in parameters:
        if key in kwargs:
            args.append(kwargs[key])
        elif param.default != param.empty:
            args.append(param.default)
        else:
            return
    return args


def _add_gradient_checkpointing(module_list):

    requires_grad = None

    def _new_forward(self, *args, **kwargs):
        nonlocal requires_grad
        if requires_grad is None:
            requires_grad = any(p.requires_grad for p in self.parameters())

        new_args = _kwargs_to_args(self.__old_forward, args, kwargs)
        if new_args is not None and self.gradient_checkpointing and self.training:
            if new_args and isinstance(new_args[0], torch.Tensor) and requires_grad and not new_args[0].requires_grad:
                new_args[0].requires_grad_(True)
            layer_ret = self._gradient_checkpointing_func(self.__old_forward, *new_args)
            logger.info_once('Successfully using dynamic gradient checkpointing.')
        else:
            layer_ret = self.__old_forward(*args, **kwargs)
        return layer_ret

    for module in module_list:
        module.gradient_checkpointing = False
        if hasattr(module, '_old_forward'):  # device_map
            __old_forward = module._old_forward
            module._old_forward = MethodType(_new_forward, module)
        else:
            __old_forward = module.forward
            module.forward = MethodType(_new_forward, module)
        module.__old_forward = __old_forward


def find_module_list(model) -> Optional[nn.ModuleList]:
    module_lists = []
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing') or m.__class__.__name__ == 'CheckpointWrapper':
            return
        if (isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) >= 10
                and 'mlp' not in m[0].__class__.__name__.lower()):  # fix moe
            module_lists.append(m)
    if module_lists:
        return max(module_lists, key=lambda x: len(x))


def dynamic_gradient_checkpointing(model, including_vit: bool = False) -> None:
    if isinstance(model, PeftModel):
        model = model.model
    model_meta: ModelMeta = getattr(model, 'model_meta', None)
    if model_meta is not None and model_meta.is_multimodal and model_meta.model_arch:
        tower_names = model_meta.model_arch.language_model.copy()
        if including_vit:
            tower_names += model_meta.model_arch.vision_tower
    else:
        tower_names = [None]

    model.supports_gradient_checkpointing = True
    for tower_name in tower_names:
        if tower_name is None:
            model_tower = model
        else:
            model_tower = deep_getattr(model, tower_name)
        model_tower.supports_gradient_checkpointing = True
        module_list = find_module_list(model_tower)
        if module_list is None:
            continue
        _add_gradient_checkpointing(module_list)
        logger.info(f'Automatically add gradient_checkpointing to {model_tower.__class__}.')


@contextmanager
def disable_gradient_checkpointing(model: PreTrainedModel, gradient_checkpointing_kwargs: Optional[Dict] = None):
    """
    Temporarily disable gradient checkpointing, restoring the previous state afterward.

    When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
    torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
    Temporarily disable checkpointing to avoid this warning during inference.

    Args:
        model (`PreTrainedModel`):
            Model for which to temporarily disable gradient checkpointing.
        gradient_checkpointing_kwargs (`dict` or `None`, *optional*):
            Additional kwargs for gradient checkpointing enabling.
    """
    was_enabled = getattr(model, 'is_gradient_checkpointing', False)
    if was_enabled:
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if was_enabled:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)


def gather_for_unpadded_tensors(input_data, use_gather_object=False):
    from accelerate.utils import gather_object
    if getattr(sequence_parallel, 'dp_group', None) is not None:
        input_data = sequence_parallel._gather_object_dp(input_data)
    else:
        input_data = gather_object(input_data)
    output = []
    for _data in input_data:
        if len(_data.shape) == 0:
            _data = _data.unsqueeze(0)
        _data = _data.cpu()
        output.append(_data)
    if len(output[0].shape) == 1 and output[0].shape[0] > 1:
        data = torch.stack(output, dim=0)
    else:
        data = torch.concat(output, dim=0)
    return data


def calculate_max_steps(args: 'TrainingArguments', dataset) -> int:
    if args.max_steps and args.max_steps > 0:
        max_steps = args.max_steps
    else:
        len_dataset = len(dataset)
        _, _, world_size, _ = get_dist_setting()
        total_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        num_update_steps_per_epoch = len_dataset // total_train_batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    return max_steps


def extract_version(name: str) -> Optional[int]:
    if not name.startswith('v'):
        return None
    try:
        num = name[1:].split('-', 1)[0]
        return int(num)
    except ValueError:
        return None


def get_previous_version_from_path(current_path: str) -> Optional[str]:
    from pathlib import Path
    current = Path(current_path)
    parent = current.parent
    current_name = current.name

    candidates = [d for d in parent.iterdir() if d.is_dir()]

    valid = [(d.name, extract_version(d.name)) for d in candidates]
    valid = [(name, ver) for name, ver in valid if ver is not None]

    valid.sort(key=lambda x: x[1])
    names = [name for name, _ in valid]

    if current_name not in names:
        return None

    idx = names.index(current_name)
    if idx == 0:
        return None

    prev_name = names[idx - 1]
    return str(parent / prev_name)


def get_resume_dir(output_dir):
    return get_previous_version_from_path(output_dir)


def replace_index_file(output_dir: str):
    from transformers.utils import WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME
    import os
    import json
    index_file = os.path.join(output_dir, WEIGHTS_INDEX_NAME)

    if not os.path.exists(index_file):
        return
    with open(index_file, 'r', encoding='utf-8') as f:
        bin_data = json.load(f)
    if 'weight_map' not in bin_data:
        return
    bin_data['weight_map'] = {
        k: v.replace('pytorch_model', 'model').replace('.bin', '.safetensors')
        for k, v in bin_data['weight_map'].items()
    }
    safe_path = os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(safe_path, 'w', encoding='utf-8') as f:
        json.dump(bin_data, f, indent=2)
    from contextlib import suppress
    with suppress(FileNotFoundError):
        os.remove(os.path.join(output_dir, WEIGHTS_INDEX_NAME))

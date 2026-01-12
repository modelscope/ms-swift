# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
from types import FunctionType, MethodType
from typing import List, Union

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.nn import CrossEntropyLoss, Module

from swift.utils import get_logger

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


def per_token_loss_func_sp(outputs, labels, enable_dft_loss=False, enable_eaft_loss=False, eaft_alpha=1.0, **kwargs) -> torch.Tensor:
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
        from swift.trainers.sequence_parallel.utils import ChunkedCrossEntropyLoss
        loss = ChunkedCrossEntropyLoss.apply(logits, labels, sploss_parallel_size)
    else:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs
    if enable_eaft_loss:
        with torch.no_grad():
            logits_detach = logits.detach()
            valid_mask = labels != -100
            logits_valid = logits_detach[valid_mask]
            
            topk_logits, topk_indices = torch.topk(logits_valid, k=20, dim=-1)
            logsumexp_topk = torch.logsumexp(topk_logits, dim=-1, keepdim=True)
            log_probs_topk = topk_logits - logsumexp_topk
            probs_topk = torch.exp(log_probs_topk)
            entropy_approx = -(probs_topk * log_probs_topk).sum(dim=-1)
            normalized_entropy = entropy_approx / 3.0
            eaft_weight_valid = torch.pow(normalized_entropy, eaft_alpha)
            
            eaft_weight = torch.ones_like(loss)
            eaft_weight[valid_mask] = eaft_weight_valid
            
        loss *= eaft_weight
    from swift.trainers.sequence_parallel import sequence_parallel
    position_ids = sequence_parallel.real_position_ids
    if position_ids is not None:
        position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
    from swift.trainers.sequence_parallel.utils import GatherLoss
    loss, labels = GatherLoss.apply(loss.reshape(batch_size, -1), labels.reshape(batch_size, -1), 1, position_ids)
    if position_ids is not None and position_ids.min() == -1:
        _pos_mask = position_ids >= 0
        loss = loss[_pos_mask].contiguous()

    return loss


def per_token_loss_func(outputs, labels, enable_dft_loss: bool = False, enable_eaft_loss: bool = False,
                        eaft_alpha: float = 1.0, **kwargs):
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
    if enable_eaft_loss:
        with torch.no_grad():
            valid_mask = labels != -100
            logits_detach = logits[valid_mask].detach()
            topk_logits, topk_indices = torch.topk(logits_detach, k=20, dim=-1)
            logsumexp_topk = torch.logsumexp(topk_logits, dim=-1, keepdim=True)
            log_probs_topk = topk_logits - logsumexp_topk
            probs_topk = torch.exp(log_probs_topk)
            entropy_approx = -(probs_topk * log_probs_topk).sum(dim=-1)
            normalized_entropy = entropy_approx / 3.0            
            eaft_weight = torch.pow(normalized_entropy, eaft_alpha)
            
            eaft_weight_full = torch.ones_like(loss)
            eaft_weight_full[valid_mask] = eaft_weight
            
        loss *= eaft_weight_full
    return loss

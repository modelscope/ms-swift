# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
from types import FunctionType, MethodType
from typing import List, Tuple, Union

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
        from swift.trainers.sequence_parallel.utils import ChunkedCrossEntropyLoss
        loss = ChunkedCrossEntropyLoss.apply(logits, labels, sploss_parallel_size)
    else:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs
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


def compute_subseq_counts_from_encoded(encoded_list: List[dict]) -> List[int]:
    """Compute per-prompt subsequence counts from pre-encoded template outputs.

    encoded_list: a list of template.encode(...) results (dicts). Returns a list of integers, one per entry.
    """
    counts = []
    for enc in encoded_list:
        pos_ids = enc.get('text_position_ids') or enc.get('position_ids')
        if pos_ids is None:
            counts.append(1)
            continue
        # pos_ids may be a 2D tensor (batch, seq) - but encoded_list entries are per sample
        pos_ids = pos_ids.squeeze()
        if pos_ids.numel() == 0:
            counts.append(1)
            continue
        start_idx = (pos_ids == 0).nonzero(as_tuple=True)[0]
        counts.append(int(start_idx.shape[0]))
    return counts


def expand_truncated_mask_to_subseq(truncated_prompt_tensor: torch.Tensor,
                                    per_prompt_counts: List[int]) -> torch.Tensor:
    """Expand per-prompt truncated mask into per-subsequence mask using counts.

    truncated_prompt_tensor: BoolTensor of shape (batch, ) indicating truncated per-prompt
    per_prompt_counts: list of ints
    Returns: BoolTensor of shape (sum(per_prompt_counts), )
    """
    if not isinstance(truncated_prompt_tensor, torch.Tensor):
        raise TypeError('truncated_prompt_tensor must be a torch.Tensor')
    per_prompt_counts_tensor = torch.tensor(per_prompt_counts, dtype=torch.long, device=truncated_prompt_tensor.device)
    return torch.repeat_interleave(truncated_prompt_tensor, per_prompt_counts_tensor)


def compute_subseq_advantages_and_counts(
    advantages: torch.Tensor,
    inputs: List[dict],
    batch_encoded_inputs: List[dict] | None,
    template,
    device: torch.device,
    allow_reencode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-subsequence advantages and subsequence counts tensor.

    - advantages: Tensor of shape (num_prompts,) per-prompt
    - inputs: List of original prompt inputs (used to re-encode if needed)
    - batch_encoded_inputs: Optional list of encoded batch dicts (may include '_per_prompt_subseq_counts')
    - template: Template object used to re-encode inputs if counts are missing
    - device: torch.device for tensor allocation

    Returns:
        (subseq_advantages, subseq_counts_tensor)
    """
    if device is None:
        device = torch.device('cpu')

    if getattr(template, 'padding_free', False):
        subseq_counts: List[int] = []
        # Use precomputed counts when available
        if batch_encoded_inputs is not None:
            for batch_encoded in batch_encoded_inputs:
                per_prompt_counts = batch_encoded.get('_per_prompt_subseq_counts')
                if per_prompt_counts is not None:
                    subseq_counts.extend(per_prompt_counts)

        # If counts are partial or missing, only re-encode if explicitly allowed
        if len(subseq_counts) != len(inputs):
            if not allow_reencode:
                raise RuntimeError('Missing per-prompt subsequence counts in batch_encoded_inputs. '
                                   'Pass pre-encoded batch (batch_encoded_inputs) or set '
                                   'allow_reencode=True to allow fallback re-encoding.')
            subseq_counts = []
            for d in inputs:
                enc = template.encode(d, return_length=True)
                pos_ids = enc.get('text_position_ids') or enc.get('position_ids')
                if pos_ids is None:
                    subseq_counts.append(1)
                else:
                    pos_ids = pos_ids.squeeze()
                    if pos_ids.numel() == 0:
                        subseq_counts.append(1)
                    else:
                        start_idx = (pos_ids == 0).nonzero(as_tuple=True)[0]
                        subseq_counts.append(int(start_idx.shape[0]))

        subseq_counts_tensor = torch.tensor(subseq_counts, dtype=torch.long, device=device)
        subseq_advantages = torch.repeat_interleave(advantages, subseq_counts_tensor)
    else:
        subseq_advantages = advantages
        subseq_counts_tensor = torch.ones((len(inputs), ), dtype=torch.long, device=device)
    return subseq_advantages, subseq_counts_tensor

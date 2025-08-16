# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Sampler

from swift.llm import DataLoaderDispatcher, DataLoaderShard, get_llm_model
from swift.utils import get_current_device, get_dist_setting, seed_worker
from .base import SequenceParallel

if TYPE_CHECKING:
    try:
        from ..rlhf_trainer import GRPOTrainer
        from ..rlhf_trainer.grpo_trainer import InputsType
    except ImportError:
        pass
# Conditional import for profiling decorator
try:
    from trl.extras.profiling import profiling_decorator
except ImportError:
    # Fallback if trl is not available
    def profiling_decorator(func):
        return func


try:
    from trl.trainer.utils import entropy_from_logits
except ImportError:
    from ..rlhf_trainer.utils import entropy_from_logits


class GatherLoss(torch.autograd.Function):
    """Gather loss from sequence group"""

    @staticmethod
    def forward(ctx, loss, labels, process_group, gather_idx=None):
        """
        Args:
            loss: loss tensor after splitting
            labels: labels tensor after splitting
            process_group: the sequence parallel group
            gather_idx: gather the tensors on this dim
        """
        ctx.process_group = process_group
        # change from label.shape to loss, because label may be None
        shape0 = loss.shape[0]
        ctx.scatter_shape = loss.shape[gather_idx or 0]
        ctx.gather_idx = gather_idx or 0
        world_size = dist.get_world_size(group=process_group)  # the sp world size
        output = torch.empty((shape0 * world_size, *loss.shape[1:]), dtype=loss.dtype, device=loss.device)
        # gather all from sp group
        dist.all_gather_into_tensor(output, loss, group=process_group)
        if gather_idx is not None:
            output = torch.cat(output.split(shape0, dim=0), dim=gather_idx)
        if labels is not None:
            labels_output = torch.empty((shape0 * world_size, *labels.shape[1:]),
                                        dtype=labels.dtype,
                                        device=labels.device)
            dist.all_gather_into_tensor(labels_output, labels, group=process_group)
            if gather_idx is not None:
                labels_output = torch.cat(labels_output.split(shape0, dim=0), dim=gather_idx)
        else:
            labels_output = None
        return output, labels_output

    @staticmethod
    def backward(ctx, *grad_output):
        _grad = grad_output[0] * dist.get_world_size(group=ctx.process_group)
        return _grad.split(
            ctx.scatter_shape, dim=ctx.gather_idx)[dist.get_rank(ctx.process_group)].contiguous(), None, None, None


class ChunkedCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, chunk_size):
        ctx.save_for_backward(logits, labels)
        ctx.chunk_size = chunk_size

        losses = []
        for i in range(math.ceil(logits.shape[0] / chunk_size)):
            l_start = i * chunk_size
            l_end = min((i + 1) * chunk_size, logits.shape[0])
            logits_chunk = logits[l_start:l_end]
            labels_chunk = labels[l_start:l_end]
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_chunk = loss_fct(logits_chunk, labels_chunk)
            losses.append(loss_chunk)
            del logits_chunk
            del labels_chunk
        all_losses = torch.cat(losses)
        return all_losses

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        logits, labels = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        for i in range(math.ceil(logits.shape[0] / chunk_size)):
            l_start = i * chunk_size
            l_end = min((i + 1) * chunk_size, logits.shape[0])
            logits_chunk = logits[l_start:l_end].detach().requires_grad_(True)
            labels_chunk = labels[l_start:l_end]
            loss_fct = CrossEntropyLoss(reduction='none')
            with torch.enable_grad():
                loss_chunk = loss_fct(logits_chunk, labels_chunk)
                grad_output_chunk = grad_outputs[0][l_start:l_end]
                _loss_chunk = (loss_chunk * grad_output_chunk).sum()
                grad_chunk = torch.autograd.grad(_loss_chunk, logits_chunk, retain_graph=False)[0]
                logits[l_start:l_end] = grad_chunk

        return logits, None, None


def loss_scale_sp_func(outputs,
                       labels,
                       loss_scale=None,
                       num_items_in_batch=None,
                       sp_instance=None,
                       enable_dft_loss=False,
                       **kwargs) -> torch.Tensor:
    """Common loss function for sequence parallel training"""
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    device = logits.device

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
    if loss_scale is not None:
        loss_scale = loss_scale.flatten().to(device)
        loss = (loss_scale * loss)
    loss, labels = GatherLoss.apply(loss, labels, sp_instance.sp_group)
    mask = (labels != -100)
    total_loss = loss[mask].sum()
    if num_items_in_batch is None:
        total_loss = total_loss / mask.sum()
    else:
        total_loss = total_loss / num_items_in_batch
    return total_loss

class SequenceParallelSampler(Sampler):
    """Sampler for sequence parallel training"""

    def __init__(self, sp_instance, dataset, shuffle: bool = True, seed=None, round_up: bool = True) -> None:
        self.sp_instance = sp_instance
        rank = dist.get_rank(sp_instance.device_mesh['data'].get_group())
        world_size = sp_instance.device_mesh['data'].size()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        assert seed is not None
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]

        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class SequenceParallelDispatcher(DataLoaderDispatcher):
    """Dispatcher for sequence parallel training"""

    def __init__(self, dataloader, sp_instance, device=None, skip_batches: int = 0):
        super().__init__(dataloader)
        self.sp_instance = sp_instance
        self.device = device
        self.skip_batches = skip_batches

    @property
    def rank(self):
        return self.sp_instance.dp_rank if dist.is_initialized() else 0

    @property
    def world_size(self):
        return self.sp_instance.dp_world_size if dist.is_initialized() else 1

    @property
    def group(self):
        return self.sp_instance.dp_group if dist.is_initialized() else 1

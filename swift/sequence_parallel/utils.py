# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import Sampler
from typing import Any, Iterator

from swift.dataloader import DataLoaderDispatcher
from .sequence_parallel import sequence_parallel


class GatherTensor(torch.autograd.Function):
    """Gather tensor from sequence group (autograd supported)"""

    @staticmethod
    def forward(ctx, tensor, dim=0, position_ids=None):
        ctx.dim = dim
        if position_ids is not None:
            position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        ctx.position_ids = position_ids
        return sequence_parallel.gather(tensor, dim=dim, position_ids=position_ids)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = sequence_parallel.split(grad_output, dim=ctx.dim, position_ids=ctx.position_ids)
        return grad_input, None, None


class GatherLoss(torch.autograd.Function):
    """Gather loss from sequence group"""

    @staticmethod
    def forward(ctx, loss, labels, gather_idx=None, position_ids=None):
        """
        Args:
            loss: loss tensor after splitting
            labels: labels tensor after splitting
            gather_idx: gather the tensors on this dim
        """
        # change from label.shape to loss, because label may be None
        ctx.scatter_shape = loss.shape[gather_idx or 0]
        ctx.gather_idx = gather_idx or 0
        if position_ids is not None:
            position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        ctx.position_ids = position_ids
        output = sequence_parallel.gather(loss, dim=ctx.gather_idx, position_ids=position_ids)
        if labels is not None:
            labels_output = sequence_parallel.gather(labels, dim=ctx.gather_idx, position_ids=position_ids)
        else:
            labels_output = None
        return output, labels_output

    @staticmethod
    def backward(ctx, *grad_output):
        _grad = grad_output[0] * sequence_parallel.world_size
        if sequence_parallel.rp_world_size > 1:
            _grad = sequence_parallel.split(_grad, dim=ctx.gather_idx, position_ids=ctx.position_ids).contiguous()
        else:
            _grad = _grad.split(
                ctx.scatter_shape, dim=ctx.gather_idx)[dist.get_rank(sequence_parallel.sp_group)].contiguous()
        return _grad, None, None, None


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

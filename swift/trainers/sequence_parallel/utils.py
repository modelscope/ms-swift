# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from functools import partial
from types import MethodType
from typing import Any, Dict, Iterator, List, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Sampler

from swift.llm import DataLoaderDispatcher, DataLoaderShard, get_llm_model, to_device
from swift.utils import get_current_device, get_device, get_dist_setting, seed_worker


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
        shape0 = labels.shape[0]
        ctx.scatter_shape = labels.shape[gather_idx or 0]
        ctx.gather_idx = gather_idx or 0
        world_size = dist.get_world_size(group=process_group)  # the sp world size
        output = torch.empty((shape0 * world_size, *loss.shape[1:]), dtype=loss.dtype, device=loss.device)
        # gather all from sp group
        dist.all_gather_into_tensor(output, loss, group=process_group)
        if gather_idx is not None:
            output = torch.cat(output.split(shape0, dim=0), dim=gather_idx)
        labels_output = torch.empty((shape0 * world_size, *labels.shape[1:]), dtype=labels.dtype, device=labels.device)
        dist.all_gather_into_tensor(labels_output, labels, group=process_group)
        if gather_idx is not None:
            labels_output = torch.cat(labels_output.split(shape0, dim=0), dim=gather_idx)
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


def loss_scale_sp_func(outputs, labels, loss_scale=None, num_items_in_batch=None, sp_instance=None) -> torch.Tensor:
    """Common loss function for sequence parallel training"""
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    device = logits.device

    if labels.shape[1] > logits.shape[1]:
        _, _, labels, _, _, loss_scale = sp_instance.pad_and_split_inputs(None, None, labels, None, None, loss_scale)
    logits = logits.view(-1, logits.shape[-1])

    labels = labels.flatten().to(device)
    sploss_parallel_size = int(os.environ.get('CELOSS_PARALLEL_SIZE', '0'))
    if sploss_parallel_size > 0:
        loss = ChunkedCrossEntropyLoss.apply(logits, labels, sploss_parallel_size)
    else:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)
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


def _prepare_inputs(self, inputs, sp_instance):
    """Common input preparation function"""
    if 'labels' in inputs:
        labels = inputs['labels']
        _, _, labels, _, _, _ = sp_instance.pad_and_split_inputs(None, None, labels, None, None, None)
        inputs['labels'] = labels
    return self._origin_prepare_inputs(inputs)


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

    def __init__(self, dataloader, sp_instance, device=None):
        super().__init__(dataloader)
        self.sp_instance = sp_instance
        self.device = device

    def _scatter_object_list(self, inputs):
        if not dist.is_initialized():
            return inputs[0]
        outputs = [None]
        global_src_rank = dist.get_global_rank(self.sp_instance.dp_group, 0)
        dist.scatter_object_list(outputs, inputs, global_src_rank, group=self.sp_instance.dp_group)
        return outputs[0]

    def __iter__(self):
        base_iter = iter(self.base_dataloader)
        while True:
            if self.sp_instance.dp_rank == 0:
                try:
                    data = [next(base_iter) for _ in range(self.sp_instance.dp_world_size)]
                except StopIteration:
                    data = [None] * self.sp_instance.dp_world_size
                data = self._scatter_object_list(data)
            else:
                data = self._scatter_object_list(None)
            if data is None:
                break
            if self.device:
                data = to_device(data, self.device)
            yield data


def setup_compute_acc(sp_instance):
    """Setup compute_acc function for sequence parallel training"""
    from swift.plugin import metric
    from swift.trainers import mixin
    compute_acc_origin = metric.compute_acc

    def compute_acc(preds, labels, *args, **kwargs) -> Dict[str, List[float]]:
        # Gather preds and labels across the sp group
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).to(get_current_device())
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(get_current_device())
        if labels.shape[1] > preds.shape[1]:
            _, _, labels, _, _, _ = sp_instance.pad_and_split_inputs(None, None, labels, None, None, None)
        shape0 = preds.shape[0]
        preds_output = torch.empty((shape0 * sp_instance.sp_world_size, preds.shape[1]),
                                   dtype=preds.dtype,
                                   device=preds.device)
        dist.all_gather_into_tensor(preds_output, preds, group=sp_instance.sp_group)
        preds_output = torch.cat(preds_output.split(shape0, dim=0), dim=1)
        shape0 = labels.shape[0]
        labels_output = torch.empty((shape0 * sp_instance.sp_world_size, labels.shape[1]),
                                    dtype=labels.dtype,
                                    device=labels.device)
        dist.all_gather_into_tensor(labels_output, labels, group=sp_instance.sp_group)
        labels_output = torch.cat(labels_output.split(shape0, dim=0), dim=1)
        # roll back to fit compute_acc
        labels_output = torch.roll(labels_output, shifts=1, dims=1)
        return compute_acc_origin(preds_output, labels_output, *args, **kwargs)

    metric.compute_acc = compute_acc
    mixin.compute_acc = compute_acc


# For DPO
def get_per_token_logps(logits: torch.FloatTensor,
                        labels: torch.LongTensor,
                        label_pad_token_id=-100,
                        sp_instance=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Common DPO per-token logps function for sequence parallel training"""
    if labels.shape[1] > logits.shape[1]:
        _, _, labels, _, _, _ = sp_instance.pad_and_split_inputs(None, None, labels, None, None, None)
    loss_mask = labels != label_pad_token_id
    labels = labels.clone()  # No need to shift, pad and split has shifted the inputs.
    labels[~loss_mask] = 0
    labels = labels.to(logits.device)
    loss_mask = loss_mask.to(logits.device)
    mean_logits = logits.mean(-1)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    total_per_token_logps, total_loss_mask = GatherLoss.apply(per_token_logps, loss_mask, sp_instance.sp_group, 1)

    world_size = dist.get_world_size(group=sp_instance.sp_group)
    total_mean_logits = mean_logits.new_empty((mean_logits.shape[0], mean_logits.shape[1] * world_size))
    dist.all_gather_into_tensor(total_mean_logits, mean_logits, group=sp_instance.sp_group)
    total_per_token_logps[~total_loss_mask] = 0
    return total_per_token_logps, total_mean_logits, total_loss_mask


def get_common_dataloader(sp_instance, trainer, dataset, batch_size, sampler_class, dispatcher_class):
    """Common dataloader creation function"""
    data_collator = trainer.data_collator
    if isinstance(dataset, datasets.Dataset):
        dataset = trainer._remove_unused_columns(dataset, description='training')
    else:
        data_collator = trainer._get_collator_with_removed_columns(data_collator, description='training')
    
    if hasattr(dataset, '__len__'):
        sampler = sampler_class(sp_instance, dataset, seed=42)
        dataloader_params = {
            'batch_size': batch_size,
            'collate_fn': data_collator,
            'num_workers': trainer.args.dataloader_num_workers,
            'pin_memory': trainer.args.dataloader_pin_memory,
            'persistent_workers': trainer.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader_params['sampler'] = sampler
            dataloader_params['drop_last'] = trainer.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = partial(
                seed_worker, num_workers=trainer.args.dataloader_num_workers, rank=trainer.args.process_index)

        return DataLoaderShard(dataset, device=trainer.accelerator.device, **dataloader_params)
    else:
        dataloader_params = {
            'collate_fn': data_collator,
            'num_workers': trainer.args.dataloader_num_workers,
            'pin_memory': trainer.args.dataloader_pin_memory,
            'persistent_workers': trainer.args.dataloader_persistent_workers,
            'prefetch_factor': trainer.args.dataloader_prefetch_factor
        }
        if dist.is_initialized() and dataloader_params['prefetch_factor']:
            dataloader_params['prefetch_factor'] = dataloader_params['prefetch_factor'] * dist.get_world_size()
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_params)
        dataloader = dispatcher_class(dataloader, sp_instance, trainer.accelerator.device)
        return dataloader 
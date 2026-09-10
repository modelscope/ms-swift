# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional

from swift.utils import to_device


class BatchSamplerShard:

    def __init__(
        self,
        total_samples: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        data_seed: Optional[int],
        tp_size: int = 1,
        group_by_length: bool = False,
        lengths=None,
    ):
        self.tp_size = tp_size
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed
        self.group_by_length = group_by_length
        if group_by_length and not shuffle:
            raise ValueError('shuffle must be True when group_by_length is True')
        self.lengths = lengths
        if self.lengths is not None:
            self.lengths = [max(length) if isinstance(length, list) else length for length in self.lengths]
        # Match DistributedSampler semantics: each DP rank must receive the same number of samples.
        if self.drop_last:
            self.num_samples = self.total_samples // self.world_size
        else:
            self.num_samples = (self.total_samples + self.world_size - 1) // self.world_size
        self.total_size = self.num_samples * self.world_size

    @property
    def rank(self):
        return (dist.get_rank() // self.tp_size) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return (dist.get_world_size() // self.tp_size) if dist.is_initialized() else 1

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            if self.group_by_length:
                from transformers.trainer_pt_utils import get_length_grouped_indices
                total_idx = get_length_grouped_indices(
                    self.lengths, self.batch_size * self.world_size, generator=generator)
            else:
                total_idx = torch.randperm(self.total_samples, generator=generator).tolist()
        else:
            total_idx = list(range(self.total_samples))

        if self.drop_last:
            total_idx = total_idx[:self.total_size]
        else:
            # Repeat from the global order so every original sample is retained, including when N < world_size.
            padding_size = self.total_size - len(total_idx)
            if padding_size > 0:
                repeats = math.ceil(padding_size / len(total_idx))
                total_idx += (total_idx * repeats)[:padding_size]
        assert len(total_idx) == self.total_size
        total_idx = total_idx[self.rank:self.total_size:self.world_size]
        assert len(total_idx) == self.num_samples

        batch = []
        # Last batch if not complete will be dropped.
        for idx in total_idx:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch
        return

    def set_epoch(self, epoch: int):
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class DataLoaderShard(DataLoader):

    def __init__(self, dataset, device=None, **dataloader_params):
        self.device = device
        super().__init__(dataset, **dataloader_params)

    def set_epoch(self, epoch: int):
        if self.batch_sampler is not None:
            if hasattr(self.batch_sampler, 'set_epoch'):
                self.batch_sampler.set_epoch(epoch)
            if hasattr(self.batch_sampler, 'batch_sampler') and hasattr(self.batch_sampler.batch_sampler, 'set_epoch'):
                self.batch_sampler.batch_sampler.set_epoch(epoch)
        elif self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        for item in super().__iter__():
            if self.device:
                item = to_device(item, self.device)
            yield item

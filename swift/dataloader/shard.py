from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

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
        self.total_samples = total_samples // self.world_size
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
                total_idx = torch.randperm(self.total_samples * self.world_size, generator=generator).tolist()
            total_idx = total_idx[self.rank::self.world_size]
        else:
            total_idx = range(self.rank, self.total_samples * self.world_size, self.world_size)

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
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


class DataLoaderShard(DataLoader):

    def __init__(self, dataset, device=None, **dataloader_params):
        self.device = device
        super().__init__(dataset, **dataloader_params)

    def set_epoch(self, epoch: int):
        if self.batch_sampler is not None and hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)
        elif self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        for item in super().__iter__():
            if self.device:
                item = to_device(item, self.device)
            yield item

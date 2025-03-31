from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


class BatchSamplerShard:

    def __init__(self, total_samples: int, batch_size: int, shuffle: bool, drop_last: bool, data_seed: Optional[int]):
        self.total_samples = total_samples // self.world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed

    @property
    def rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1

    def __iter__(self):
        start_idx = self.rank * self.total_samples
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            total_idx = torch.randperm(self.total_samples * self.world_size, generator=generator).tolist()
            total_idx = total_idx[start_idx:start_idx + self.total_samples]
        else:
            total_idx = list(range(start_idx, start_idx + self.total_samples))

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

    def __init__(self, dataset, batch_sampler: BatchSamplerShard, **dataloader_params):
        self.batch_sampler = batch_sampler
        super().__init__(dataset, batch_sampler=self.batch_sampler, **dataloader_params)

    def set_epoch(self, epoch: int):
        self.batch_sampler.set_epoch(epoch)


class DataLoaderDispatcher:

    def __init__(self, base_dataloader):
        self.base_dataloader = base_dataloader

    @property
    def rank(self):
        return dist.get_rank(self.group) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size(self.group) if dist.is_initialized() else 1

    @property
    def group(self):
        return dist.group.WORLD if dist.is_initialized() else 1

    @property
    def src_rank(self):
        return 0

    def _scatter_object_list(self, inputs):
        if not dist.is_initialized():
            return inputs[0]
        outputs = [None]
        dist.scatter_object_list(outputs, inputs, self.src_rank, group=self.group)
        return outputs[0]

    def __iter__(self):
        base_iter = iter(self.base_dataloader)
        while True:
            if self.rank == self.src_rank:
                try:
                    data = [next(base_iter) for _ in range(self.world_size)]
                except StopIteration:
                    data = [None] * self.world_size
                data = self._scatter_object_list(data)
            else:
                data = self._scatter_object_list(None)
            if data is None:
                break
            yield data

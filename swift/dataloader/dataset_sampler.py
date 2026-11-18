# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
from collections import defaultdict

from .shard import BatchSamplerShard


class DatasetBatchSamplerShard(BatchSamplerShard):
    """Shuffle global batches, then shard each batch without mixing source datasets."""

    def __init__(self, groups, batch_size, **kwargs):
        if kwargs.pop('group_by_length', False):
            raise ValueError('Dataset batching does not support group_by_length.')
        kwargs.pop('lengths', None)
        super().__init__(len(groups), batch_size, **kwargs)
        self.groups = defaultdict(list)
        for index, group in enumerate(groups):
            if group is None:
                raise ValueError('Every sample requires a dataset group when dataset batching is enabled.')
            self.groups[group].append(index)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.curr_seed)
        global_size = self.batch_size * self.world_size
        batches = []
        for indices in self.groups.values():
            if self.shuffle:
                indices = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
            else:
                indices = indices.copy()
            remainder = len(indices) % global_size
            if remainder:
                if self.drop_last:
                    indices = indices[:-remainder]
                else:
                    # Pad within this dataset so all ranks have equally sized batches and step counts.
                    needed = global_size - remainder
                    indices += (indices * math.ceil(needed / len(indices)))[:needed]
            batches.extend(indices[i:i + global_size] for i in range(0, len(indices), global_size))
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
        for batch in batches:
            yield batch[self.rank::self.world_size]

    def __len__(self):
        global_size = self.batch_size * self.world_size
        return sum((len(indices) // global_size if self.drop_last else math.ceil(len(indices) / global_size))
                   for indices in self.groups.values())

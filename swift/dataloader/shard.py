# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

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


class DynamicMixBatchSampler:
    """Batch sampler that samples indices weighted by per-domain probabilities.

    Supports runtime probability updates for dynamic data mixing.
    """

    def __init__(
        self,
        domain_indices: Dict[str, List[int]],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        data_seed: Optional[int],
        tp_size: int = 1,
        num_batches: Optional[int] = None,
    ):
        self.tp_size = tp_size
        self.domain_indices = domain_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed
        self.num_batches = num_batches
        # Sort domain names to ensure consistent ordering across all ranks
        self.domain_names = sorted(domain_indices.keys())
        # Initial weights proportional to domain sizes
        total = sum(len(domain_indices[n]) for n in self.domain_names)
        self.probabilities = {n: len(domain_indices[n]) / total for n in self.domain_names}

    @property
    def rank(self):
        return (dist.get_rank() // self.tp_size) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return (dist.get_world_size() // self.tp_size) if dist.is_initialized() else 1

    def set_probabilities(self, probs):
        """Update sampling probabilities (must be called with the same values on all ranks)."""
        for name in self.domain_names:
            if name in probs:
                self.probabilities[name] = probs[name]
        # Normalize
        total = sum(self.probabilities[n] for n in self.domain_names)
        self.probabilities = {n: self.probabilities[n] / total for n in self.domain_names}

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.curr_seed)
        # Shuffle indices within each domain
        domain_shuffled = {}
        for name in self.domain_names:
            indices = self.domain_indices[name]
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                domain_shuffled[name] = [indices[p] for p in perm]
            else:
                domain_shuffled[name] = list(indices)
        domain_cursors = {name: 0 for name in self.domain_names}

        for _ in range(self.num_batches):
            # Re-read probabilities each batch so runtime updates take effect
            prob_tensor = torch.tensor([self.probabilities[n] for n in self.domain_names])
            global_batch = []
            for _ in range(self.batch_size * self.world_size):
                domain_idx = torch.multinomial(prob_tensor, 1, generator=generator).item()
                domain_name = self.domain_names[domain_idx]
                cursor = domain_cursors[domain_name]
                if cursor >= len(domain_shuffled[domain_name]):
                    # Domain exhausted, reshuffle and reset
                    indices = self.domain_indices[domain_name]
                    if self.shuffle:
                        perm = torch.randperm(len(indices), generator=generator).tolist()
                        domain_shuffled[domain_name] = [indices[p] for p in perm]
                    domain_cursors[domain_name] = 0
                    cursor = 0
                global_batch.append(domain_shuffled[domain_name][cursor])
                domain_cursors[domain_name] = cursor + 1
            # Distributed sharding: each rank takes its slice
            yield global_batch[self.rank::self.world_size]

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __len__(self):
        return self.num_batches

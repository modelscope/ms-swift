# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from swift.utils import get_logger

logger = get_logger()


# Code borrowed from megatron-lm
class MegatronPretrainingSampler:

    def __init__(self,
                 total_samples,
                 consumed_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size,
                 drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


# Code borrowed from megatron-lm
class MegatronPretrainingRandomSampler:

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
        shuffle: bool = True,
        group_by_length: bool = False,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        if group_by_length:
            if data_sharding:
                data_sharding = False
                logger.warning('`group_by_length=True` is incompatible with `data_sharding=True`. '
                               'Setting `data_sharding=False` to enable length grouping.')
            if not shuffle:
                raise ValueError('shuffle must be True when group_by_length is True')
        self.data_sharding = data_sharding
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.lengths = self.dataset['lengths'] if group_by_length else None
        if self.lengths is not None:
            self.lengths = [max(length) if isinstance(length, list) else length for length in self.lengths]
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, (
            'data_parallel_rank should be smaller than data size: {}, '
            '{}'.format(self.data_parallel_rank, data_parallel_size))

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if self.shuffle:
            # data sharding and random sampling
            if self.data_sharding:
                bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
                bucket_offset = current_epoch_samples // self.data_parallel_size
                start_idx = self.data_parallel_rank * bucket_size

                g = torch.Generator()
                g.manual_seed(self.epoch)
                random_idx = torch.randperm(bucket_size, generator=g).tolist()
                idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
            else:
                full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
                full_bucket_offset = current_epoch_samples
                g = torch.Generator()
                g.manual_seed(self.epoch)
                if self.group_by_length:
                    from transformers.trainer_pt_utils import get_length_grouped_indices
                    idx_range_total = get_length_grouped_indices(
                        self.lengths, self.micro_batch_times_data_parallel_size, generator=g)
                else:
                    idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
                idx_range_active = idx_range_total[full_bucket_offset:]
                idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            idx_range = range(full_bucket_offset + self.data_parallel_rank, full_bucket_size, self.data_parallel_size)

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

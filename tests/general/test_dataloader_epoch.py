# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from accelerate.data_loader import SkipBatchSampler
from torch.utils.data import DistributedSampler

from swift.dataloader import BatchSamplerShard, DataLoaderShard


class TestDataLoaderEpoch(unittest.TestCase):

    def test_ordinary_sampler_reshuffles(self):
        dataset = list(range(16))
        for rank in (0, 1):
            with self.subTest(rank=rank):
                sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42)
                reference = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42)
                loader = DataLoaderShard(dataset, batch_size=2, sampler=sampler)
                orders = []
                for epoch in (0, 1, 2):
                    loader.set_epoch(epoch)
                    reference.set_epoch(epoch)
                    order = [item for batch in loader for item in batch.tolist()]
                    self.assertEqual(sampler.epoch, epoch)
                    self.assertEqual(order, list(reference))
                    orders.append(order)
                self.assertNotEqual(orders[0], orders[1])

    def test_batch_sampler_and_resume_wrapper(self):
        dataset = list(range(16))
        for skip_batches in (0, 2):
            with self.subTest(skip_batches=skip_batches):
                sampler = BatchSamplerShard(16, batch_size=2, shuffle=True, drop_last=False, data_seed=42)
                reference = BatchSamplerShard(16, batch_size=2, shuffle=True, drop_last=False, data_seed=42)
                batch_sampler = SkipBatchSampler(sampler, skip_batches=skip_batches) if skip_batches else sampler
                loader = DataLoaderShard(dataset, batch_sampler=batch_sampler)
                for epoch in (0, 1, 2):
                    loader.set_epoch(epoch)
                    reference.set_epoch(epoch)
                    self.assertEqual(sampler.curr_seed, 42 + epoch)
                    self.assertEqual([batch.tolist() for batch in loader], list(reference)[skip_batches:])

    def test_unbatched_sampler(self):
        dataset = list(range(16))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=42)
        reference = DistributedSampler(dataset, num_replicas=1, rank=0, seed=42)
        loader = DataLoaderShard(dataset, batch_size=None, sampler=sampler)
        loader.set_epoch(2)
        reference.set_epoch(2)
        self.assertEqual(sampler.epoch, 2)
        self.assertEqual(list(loader), list(reference))

    def test_sampler_without_epoch_support(self):
        loader = DataLoaderShard(list(range(8)), batch_size=2)
        loader.set_epoch(1)
        self.assertEqual([item for batch in loader for item in batch.tolist()], list(range(8)))


if __name__ == '__main__':
    unittest.main()

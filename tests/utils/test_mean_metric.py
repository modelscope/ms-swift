# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
from datetime import timedelta

from swift.metrics import MeanMetric


def _mean_metric_worker(rank, init_file):
    dist.init_process_group(
        'gloo', init_method=f'file://{init_file}', rank=rank, world_size=2, timeout=timedelta(seconds=30))
    failures = []

    def check(metric, expected, label):
        local_state = (metric.state, metric.count)
        actual = metric.compute()['value']
        if abs(actual - expected) > 1e-6:
            failures.append(f'{label}: expected {expected}, got {actual}')
        if (metric.state, metric.count) != local_state:
            failures.append(f'{label}: compute changed local accumulators')

    try:
        for group in (None, dist.new_group([0, 1])):
            metric = MeanMetric(nan_value=-1, device='cpu', group=group)
            metric.update(1. if rank == 0 else 3.)
            check(metric, 2., 'initial mean')
            check(metric, 2., 'repeated compute')
            metric.update(5. if rank == 0 else 7.)
            check(metric, 4., 'update after compute')

            metric.reset()
            check(metric, -1., 'reset')
            if rank == 0:
                metric.update([2., 4.])
            check(metric, 3., 'one empty rank')
            check(metric, 3., 'one empty rank repeated compute')
            if rank == 1:
                metric.update(6.)
            check(metric, 4., 'previously empty rank receives data')
        assert not failures, '\n'.join(failures)
    finally:
        dist.destroy_process_group()


class TestMeanMetric(unittest.TestCase):

    def test_local_compute_preserves_accumulators(self):
        metric = MeanMetric(device='cpu')
        metric.update(torch.tensor([1., 3.]))
        self.assertEqual(metric.compute(), {'value': 2.})
        self.assertEqual(metric.compute(), {'value': 2.})
        self.assertEqual((metric.state, metric.count), (4., 2))
        metric.update(5.)
        self.assertEqual(metric.compute(), {'value': 3.})
        metric.reset()
        self.assertEqual((metric.state, metric.count), (0., 0))
        self.assertEqual(metric.compute(), {'value': 0})

    def test_custom_empty_value(self):
        metric = MeanMetric(nan_value=None, device='cpu')
        self.assertEqual(metric.compute(), {'value': None})
        metric.update([1., 3.])
        metric.reset()
        self.assertEqual(metric.compute(), {'value': None})

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is required')
    def test_distributed_compute_preserves_accumulators(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_mean_metric_worker, args=(os.path.join(directory, 'init'), ), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()

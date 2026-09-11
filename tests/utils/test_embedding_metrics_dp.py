# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
from datetime import timedelta
from unittest import mock

from swift.metrics.embedding import InfonceMetrics, PairedMetrics


def _shard(kind, rank):
    generator = torch.Generator().manual_seed(42 + rank)
    count = rank + 2
    if kind == 'paired':
        return torch.randn(count * 2, 8, generator=generator), torch.arange(count).float() + rank
    predictions, labels = [], []
    for i in range(count):
        negatives = 1 + (i + rank) % 3
        predictions.append(torch.randn(negatives + 2, 8, generator=generator))
        labels.extend([1] + [0] * negatives)
    return torch.cat(predictions), torch.tensor(labels)


def _metric_worker(rank, init_file):
    dist.init_process_group(
        'gloo', init_method=f'file://{init_file}', rank=rank, world_size=4, timeout=timedelta(seconds=90))
    groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    singletons = [dist.new_group([r]) for r in range(4)]
    peers = [0, 2] if rank % 2 == 0 else [1, 3]
    group = groups[rank % 2]
    failures = []
    try:
        for kind, cls in [('paired', PairedMetrics), ('infonce', InfonceMetrics)]:
            local = cls(None, None)
            local.update(*_shard(kind, rank))
            expected_local = local.compute()
            local.group = singletons[rank]
            assert local.compute() == expected_local
            for empty_peer in [False, True]:
                metric = cls(None, None)
                metric.group = group
                included = peers[1:] if empty_peer else peers
                if rank in included:
                    metric.update(*_shard(kind, rank))
                shards = [_shard(kind, p) for p in included]
                reference = cls(None, None)
                reference.update(torch.cat([s[0] for s in shards]), torch.cat([s[1] for s in shards]))
                expected = reference.compute()
                original_count = len(metric.labels)
                for repeat in range(2):
                    try:
                        actual = metric.compute()
                        for key in expected:
                            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6)
                        assert len(metric.labels) == original_count
                    except (AssertionError, ValueError) as error:
                        failures.append(f'{kind} empty_peer={empty_peer} repeat={repeat}: {error}')
                metric.reset()
                assert not metric.labels and not metric.last_hidden_state
        assert not failures, '\n'.join(failures)
    finally:
        dist.destroy_process_group()


class TestEmbeddingMetricsDataParallel(unittest.TestCase):

    def setUp(self):
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in ['INFONCE_USE_BATCH', 'INFONCE_HARD_NEGATIVES']:
            os.environ.pop(name, None)

    def test_local_compute_matches_hf_entry(self):
        from transformers import EvalPrediction
        for kind, cls in [('paired', PairedMetrics), ('infonce', InfonceMetrics)]:
            predictions, labels = _shard(kind, 0)
            metric = cls(None, None)
            metric.update(predictions, labels)
            expected = metric.compute_metrics(EvalPrediction(predictions.numpy(), labels.numpy()))
            self.assertEqual(metric.compute(), expected)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is required')
    def test_independent_data_parallel_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_metric_worker, args=(os.path.join(directory, 'init'), ), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
from datetime import timedelta
from transformers import EvalPrediction
from unittest import mock

from swift.metrics.reranker import RerankerMetrics


def _update_ranks(metric, ranks):
    for rank in ranks:
        metric.update(torch.tensor([0.] + [1.] * (rank - 1)), torch.tensor([1] + [0] * (rank - 1)))


def _distributed_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        groups = [dist.new_group(ranks, timeout=timedelta(seconds=60)) for ranks in ([0, 1], [2, 3])]
        singletons = [dist.new_group([r], timeout=timedelta(seconds=60)) for r in range(4)]
        dist.barrier()
        group = groups[rank // 2]
        check = unittest.TestCase()
        with mock.patch('swift.metrics.utils.get_current_device', return_value='cpu'):
            metric = RerankerMetrics(None, None, group=group)
            # Rank 0 has one valid query; rank 1 has two. Singleton queries are skipped.
            query_ranks = [[2], [3, 3], [2], [4]][rank]
            _update_ranks(metric, query_ranks)
            _update_ranks(metric, [1])
            expected_ranks = [2, 3, 3] if rank < 2 else [2, 4]
            with mock.patch.object(dist, 'all_reduce', wraps=dist.all_reduce) as reduce:
                result = metric.compute()
            check.assertEqual(reduce.call_count, 2)
            for call in reduce.call_args_list:
                check.assertEqual(call.args[0].numel(), 2)
                check.assertIs(call.kwargs['group'], group)
            check.assertAlmostEqual(result['mrr'], sum(1 / r for r in expected_ranks) / len(expected_ranks), places=6)
            check.assertAlmostEqual(
                result['ndcg'], sum(1 / math.log2(r + 1) for r in expected_ranks) / len(expected_ranks), places=6)
            # Computing again must not re-reduce already global state.
            check.assertEqual(metric.compute(), result)

            for empty_update in (False, True):
                metric.reset()
                if rank % 2:
                    _update_ranks(metric, [2])
                elif empty_update:
                    _update_ranks(metric, [1])
                result = metric.compute()
                check.assertAlmostEqual(result['mrr'], 0.5)
                check.assertAlmostEqual(result['ndcg'], 1 / math.log2(3), places=6)

            metric.reset()
            check.assertEqual(metric.compute(), {'mrr': 0., 'ndcg': 0.})
            metric = RerankerMetrics(None, None, group=singletons[rank])
            _update_ranks(metric, [rank + 2])
            check.assertAlmostEqual(metric.compute()['mrr'], 1 / (rank + 2), places=6)

            # The default/HF path must remain local even with a process group initialized.
            local = RerankerMetrics(None, None)
            _update_ranks(local, [2])
            with mock.patch.object(dist, 'all_reduce', side_effect=AssertionError('unexpected collective')):
                check.assertAlmostEqual(local.compute()['mrr'], 0.5)
        # Keep the default group alive until all subgroups have finished.
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestRerankerMetrics(unittest.TestCase):

    def test_local_metrics_and_reset(self):
        metric = RerankerMetrics(None, None)
        _update_ranks(metric, [2, 3, 1])
        result = metric.compute()
        self.assertAlmostEqual(result['mrr'], (0.5 + 1 / 3) / 2)
        self.assertAlmostEqual(result['ndcg'], (1 / math.log2(3) + 0.5) / 2)
        metric.reset()
        self.assertEqual(metric.compute(), {'mrr': 0., 'ndcg': 0.})

    def test_query_split_across_updates(self):
        metric = RerankerMetrics(None, None)
        metric.update(torch.tensor([0.]), torch.tensor([1]))
        metric.update(torch.tensor([1., 2.]), torch.tensor([0, 0]))
        self.assertAlmostEqual(metric.compute()['mrr'], 1 / 3)

    def test_hf_compute_metrics_stays_local(self):
        metric = RerankerMetrics(None, None)
        prediction = EvalPrediction(predictions=torch.tensor([0., 1.]), label_ids=torch.tensor([1, 0]))
        with mock.patch.object(dist, 'all_reduce', side_effect=AssertionError('unexpected collective')):
            self.assertAlmostEqual(metric.compute_metrics(prediction)['mrr'], 0.5)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is required')
    def test_data_parallel_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = 'file://' + os.path.join(directory, 'rendezvous')
            mp.spawn(_distributed_worker, args=(rendezvous, ), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()

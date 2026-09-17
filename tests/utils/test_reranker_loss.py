# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import unittest
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel
from types import SimpleNamespace
from unittest.mock import patch

from swift.loss.reranker import ListwiseRerankerLoss


def _ddp_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        torch.manual_seed(42)
        model = DistributedDataParallel(torch.nn.Linear(2, 1, bias=False))
        loss_func = ListwiseRerankerLoss(None, None)
        inputs = torch.tensor([[1., 2.], [3., 1.], [-1., 0.]])
        with patch.dict(os.environ, {'LISTWISE_RERANKER_MIN_GROUP_SIZE': '3', 'LISTWISE_RERANKER_TEMPERATURE': '1.0'}):
            # Include all-skipped, retained and uneven steps in the same DDP model.
            for counts in [(2, 2), (3, 3), (2, 3)]:
                model.zero_grad(set_to_none=True)
                count = counts[rank]
                logits = model(inputs[:count])
                loss = loss_func(SimpleNamespace(logits=logits), torch.tensor([1] + [0] * (count - 1)))
                loss.backward()
                if max(counts) == 2:
                    torch.testing.assert_close(model.module.weight.grad, torch.zeros_like(model.module.weight))
                    continue
                expected = model.module.weight.detach().clone().requires_grad_()
                scores = F.linear(inputs, expected).squeeze(-1)
                reference_loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([0]))
                reference_loss.backward()
                reference_grad = expected.grad * (sum(c == 3 for c in counts) / 2)
                torch.testing.assert_close(model.module.weight.grad, reference_grad)
            # Flush accumulated gradients even when the final local microbatch is skipped.
            for first_counts in [(2, 3), (3, 2)]:
                model.zero_grad(set_to_none=True)
                with model.no_sync():
                    count = first_counts[rank]
                    logits = model(inputs[:count])
                    loss = loss_func(SimpleNamespace(logits=logits), torch.tensor([1] + [0] * (count - 1)))
                    (loss / 2).backward()
                count = 5 - first_counts[rank]
                logits = model(inputs[:count])
                loss = loss_func(SimpleNamespace(logits=logits), torch.tensor([1] + [0] * (count - 1)))
                (loss / 2).backward()
                expected = model.module.weight.detach().clone().requires_grad_()
                F.cross_entropy(F.linear(inputs, expected).T, torch.tensor([0])).backward()
                torch.testing.assert_close(model.module.weight.grad, expected.grad / 2)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestListwiseRerankerLoss(unittest.TestCase):

    def setUp(self):
        self.loss_func = ListwiseRerankerLoss(None, None)

    def test_skipped_groups_have_zero_gradients(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            for labels in ([0, 0], [1, 0]):
                with self.subTest(
                        dtype=dtype, labels=labels), patch.dict(os.environ, {'LISTWISE_RERANKER_MIN_GROUP_SIZE': '3'}):
                    logits = torch.tensor([[1.], [2.]], dtype=dtype, requires_grad=True)
                    loss = self.loss_func(SimpleNamespace(logits=logits), torch.tensor(labels))
                    self.assertEqual(loss.item(), 0.)
                    loss.backward()
                    self.assertIsNotNone(logits.grad)
                    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_skipped_nonfinite_scores_do_not_contaminate_zero(self):
        logits = torch.tensor([[float('nan')], [float('inf')]], requires_grad=True)
        loss = self.loss_func(SimpleNamespace(logits=logits), torch.tensor([0, 0]))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_mixed_groups_preserve_retained_loss_and_gradients(self):
        with patch.dict(os.environ, {'LISTWISE_RERANKER_MIN_GROUP_SIZE': '3', 'LISTWISE_RERANKER_TEMPERATURE': '0.7'}):
            logits = torch.tensor([[4.], [2.], [1.], [3.], [2.]], dtype=torch.float64, requires_grad=True)
            loss = self.loss_func(SimpleNamespace(logits=logits), torch.tensor([1, 0, 1, 0, 0]))
            reference_logits = logits.detach().clone().requires_grad_()
            reference = F.cross_entropy(reference_logits[2:].T / 0.7, torch.tensor([0]))
            torch.testing.assert_close(loss, reference)
            loss.backward()
            reference.backward()
            torch.testing.assert_close(logits.grad, reference_logits.grad)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is required')
    def test_skipped_steps_and_uneven_ranks_under_ddp(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_ddp_worker, args=('file://' + os.path.join(directory, 'store'), ), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()

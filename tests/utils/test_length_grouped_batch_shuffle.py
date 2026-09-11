import importlib.util
import torch
import unittest
from accelerate.data_loader import SkipBatchSampler
from pathlib import Path
from transformers.trainer_pt_utils import get_length_grouped_indices
from unittest.mock import patch

from swift.dataloader.shard import BatchSamplerShard
from swift.dataloader.utils import shuffle_batch_blocks

# Load the sampler without requiring Megatron's optional training dependencies.
_spec = importlib.util.spec_from_file_location('_megatron_batch_sampler',
                                               Path(__file__).parents[2] / 'swift/megatron/trainers/batch_sampler.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
MegatronPretrainingRandomSampler = _module.MegatronPretrainingRandomSampler


class TestLengthGroupedBatchShuffle(unittest.TestCase):

    def test_preserves_blocks_tail_and_longest_first(self):
        for size in [0, 1, 7, 8, 9, 16, 17, 24, 249]:
            with self.subTest(size=size):
                lengths = [(i * 73) % 997 + 1 for i in range(size)]
                indices = get_length_grouped_indices(lengths, 8) if size else []
                original = list(indices)
                shuffled = shuffle_batch_blocks(indices, 8, torch.Generator().manual_seed(42))
                full_end = size // 8 * 8

                def blocks(x):
                    return sorted(tuple(x[i:i + 8]) for i in range(0, full_end, 8))

                self.assertEqual(blocks(original), blocks(shuffled))
                self.assertEqual(shuffled[:8], original[:8])
                self.assertEqual(shuffled[full_end:], original[full_end:])
                self.assertEqual(sorted(shuffled), list(range(size)))
                self.assertEqual(indices, original)
                if size:
                    self.assertEqual(lengths[shuffled[0]], max(lengths))

    def test_deterministic_and_breaks_sorted_order(self):
        indices = list(range(400))
        shuffled = shuffle_batch_blocks(indices, 4, torch.Generator().manual_seed(42))
        self.assertNotEqual(shuffled, indices)
        self.assertEqual(shuffled, shuffle_batch_blocks(indices, 4, torch.Generator().manual_seed(42)))
        self.assertNotEqual(shuffled, shuffle_batch_blocks(indices, 4, torch.Generator().manual_seed(43)))

    @staticmethod
    def shard_batches(lengths, rank, dp_size, enabled, epoch=0, drop_last=False, tp_size=1, skip=0, grouped=True):
        with patch(
                'torch.distributed.is_initialized', return_value=True), patch(
                    'torch.distributed.get_world_size', return_value=dp_size * tp_size), patch(
                        'torch.distributed.get_rank', return_value=rank * tp_size):
            sampler = BatchSamplerShard(
                len(lengths),
                2,
                True,
                drop_last,
                42,
                tp_size=tp_size,
                group_by_length=grouped,
                lengths=lengths,
                group_by_length_shuffle_batches=enabled)
            sampler.set_epoch(epoch)
            return list(SkipBatchSampler(sampler, skip_batches=skip))

    def test_distributed_blocks_resume_epochs_and_tp(self):
        for dp_size in [1, 2, 4, 8, 56]:
            for tail in [0, 1, dp_size * 2 - 1]:
                for drop_last in [False, True]:
                    with self.subTest(dp_size=dp_size, tail=tail, drop_last=drop_last):
                        lengths = [(i * 73) % 997 + 1 for i in range(80 * dp_size + tail)]
                        before = [
                            self.shard_batches(lengths, r, dp_size, False, drop_last=drop_last) for r in range(dp_size)
                        ]
                        after = [
                            self.shard_batches(lengths, r, dp_size, True, drop_last=drop_last) for r in range(dp_size)
                        ]

                        # A block contains the ordered local batch for every rank. Equality proves
                        # both local padding and cross-rank load balance are unchanged.
                        def blocks(ranks):
                            return sorted(tuple(tuple(b) for b in step) for step in zip(*ranks))

                        self.assertEqual(blocks(before), blocks(after))
                        for rank in range(dp_size):
                            self.assertEqual(sorted(sum(before[rank], [])), sorted(sum(after[rank], [])))
                            self.assertEqual(
                                after[rank],
                                self.shard_batches(lengths, rank, dp_size, True, drop_last=drop_last, tp_size=2))
                            self.assertEqual(
                                after[rank][7:],
                                self.shard_batches(lengths, rank, dp_size, True, drop_last=drop_last, skip=7))
                            epoch1 = self.shard_batches(lengths, rank, dp_size, True, epoch=1, drop_last=drop_last)
                            self.assertNotEqual(epoch1, after[rank])
                            self.assertEqual(
                                epoch1[7:],
                                self.shard_batches(lengths, rank, dp_size, True, epoch=1, drop_last=drop_last, skip=7))

    def test_default_and_ungrouped_behavior(self):
        lengths = list(range(320))
        expected = get_length_grouped_indices(lengths, 8, generator=torch.Generator().manual_seed(42))
        for rank in range(4):
            actual = sum(self.shard_batches(lengths, rank, 4, False), [])
            self.assertEqual(actual, expected[rank::4])
            self.assertEqual(
                self.shard_batches(lengths, rank, 4, False, grouped=False),
                self.shard_batches(lengths, rank, 4, True, grouped=False))

    def test_megatron_resume_and_blocks(self):
        for dp_size in [1, 2, 4, 8, 56]:
            for tail in [0, 1]:
                with self.subTest(dp_size=dp_size, tail=tail):
                    lengths = [(i * 73) % 997 + 1 for i in range(80 * dp_size + tail)]
                    active = len(lengths) // (2 * dp_size) * (2 * dp_size)

                    def batches(rank, enabled, consumed=0):
                        return list(
                            MegatronPretrainingRandomSampler({'lengths': lengths},
                                                             len(lengths),
                                                             consumed,
                                                             2,
                                                             rank,
                                                             dp_size,
                                                             False,
                                                             group_by_length=True,
                                                             group_by_length_shuffle_batches=enabled))

                    before = [batches(r, False) for r in range(dp_size)]
                    after = [batches(r, True) for r in range(dp_size)]

                    def blocks(ranks):
                        return sorted(tuple(tuple(b) for b in step) for step in zip(*ranks))

                    self.assertEqual(blocks(before), blocks(after))
                    for rank in range(dp_size):
                        self.assertEqual(after[rank][7:], batches(rank, True, 7 * 2 * dp_size))
                        epoch1 = batches(rank, True, active)
                        self.assertNotEqual(epoch1, after[rank])
                        self.assertEqual(epoch1[7:], batches(rank, True, active + 7 * 2 * dp_size))


if __name__ == '__main__':
    unittest.main()

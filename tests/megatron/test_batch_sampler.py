# Copyright (c) ModelScope Contributors. All rights reserved.
import runpy
import unittest
from pathlib import Path

# The sampler itself only needs PyTorch; avoid initializing the Megatron backend.
MegatronPretrainingRandomSampler = runpy.run_path(
    str(Path(__file__).resolve().parents[2]
        / 'swift/megatron/trainers/batch_sampler.py'))['MegatronPretrainingRandomSampler']


def make_sampler(total, micro_batch_size, rank, dp_size, mode, consumed=0):
    return MegatronPretrainingRandomSampler(
        dataset={'lengths': [(index * 7) % 31 + 1 for index in range(total)]},
        total_samples=total,
        consumed_samples=consumed,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=rank,
        data_parallel_size=dp_size,
        data_sharding=mode == 'sharded',
        shuffle=mode != 'sequential',
        group_by_length=mode == 'grouped',
        seed=42)


class TestMegatronBatchSampler(unittest.TestCase):

    def test_training_ranks_consume_equal_full_batches(self):
        for mode in ('sequential', 'random', 'grouped', 'sharded'):
            for total, micro_batch_size, dp_size in ((6, 1, 4), (10, 2, 3), (25, 3, 4), (12, 2, 3)):
                with self.subTest(mode=mode, total=total, micro_batch_size=micro_batch_size, dp_size=dp_size):
                    global_micro_batch = micro_batch_size * dp_size
                    batches_per_rank = total // global_micro_batch
                    active_samples = batches_per_rank * global_micro_batch
                    samplers = [make_sampler(total, micro_batch_size, rank, dp_size, mode) for rank in range(dp_size)]

                    for epoch in range(3):
                        rank_batches = [list(sampler) for sampler in samplers]
                        assert [len(batches) for batches in rank_batches] == [batches_per_rank] * dp_size
                        assert [sampler.epoch for sampler in samplers] == [epoch] * dp_size
                        assert [sampler.consumed_samples
                                for sampler in samplers] == [(epoch + 1) * active_samples] * dp_size
                        assert all(len(batch) == micro_batch_size for batches in rank_batches for batch in batches)
                        indices = [index for batches in rank_batches for batch in batches for index in batch]
                        assert len(set(indices)) == active_samples
                        assert all(0 <= index < total for index in indices)

    def test_resume_matches_remaining_epoch_batches(self):
        for mode in ('sequential', 'random', 'grouped', 'sharded'):
            for epoch in (0, 2):
                for completed_batches in (1, 2):
                    with self.subTest(mode=mode, epoch=epoch, completed_batches=completed_batches):
                        total, micro_batch_size, dp_size = 22, 2, 3
                        global_micro_batch = micro_batch_size * dp_size
                        active_samples = total // global_micro_batch * global_micro_batch
                        for rank in range(dp_size):
                            start = epoch * active_samples
                            uninterrupted = make_sampler(total, micro_batch_size, rank, dp_size, mode, consumed=start)
                            batches = list(uninterrupted)
                            resumed = make_sampler(
                                total,
                                micro_batch_size,
                                rank,
                                dp_size,
                                mode,
                                consumed=start + completed_batches * global_micro_batch)
                            remaining = list(resumed)
                            assert len(remaining) == active_samples // global_micro_batch - completed_batches
                            assert remaining == batches[completed_batches:]
                            assert resumed.consumed_samples == (epoch + 1) * active_samples


if __name__ == '__main__':
    unittest.main()

# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import time
import torch
import torch.distributed as dist
import unittest
from datetime import timedelta
from torch.utils.data import DataLoader, IterableDataset

from swift.dataloader.dispatcher import DataLoaderDispatcher, compute_streaming_skip_batches


class _FiniteEpochStream(IterableDataset):

    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        yield from range(100 * self.epoch, 100 * self.epoch + 8)


def _gloo_resume_worker(rank, init_method):
    dist.init_process_group('gloo', init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        uninterrupted = DataLoaderDispatcher(DataLoader(_FiniteEpochStream(), batch_size=None))
        epoch_zero = list(uninterrupted)
        uninterrupted.set_epoch(1)
        epoch_one = list(uninterrupted)
        assert epoch_zero == list(range(rank, 8, 2))
        assert epoch_one == list(range(100 + rank, 108, 2))

        # Adapted from AlkaidLu's two-rank regression test linked in PR #10520.
        # Each pass holds four logical batches across the two ranks.
        for offset in (2, 4, 5, 9):
            resumed = DataLoaderDispatcher(DataLoader(_FiniteEpochStream(), batch_size=None), skip_batches=offset)
            first_pass = list(resumed)
            resumed.set_epoch(1)
            second_pass = list(resumed)
            assert first_pass == epoch_zero[offset:]
            assert second_pass == epoch_one
            if offset <= 4:
                assert first_pass + second_pass == (epoch_zero + epoch_one)[offset:]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='requires PyTorch Gloo')
def test_two_rank_gloo_streaming_resume(tmp_path):
    """Check both ranks at, within and beyond the first finite pass."""
    init_method = (tmp_path / 'gloo_init').as_uri()
    workers = torch.multiprocessing.spawn(_gloo_resume_worker, args=(init_method, ), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not workers.join(timeout=max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                pytest.fail('The two Gloo ranks did not finish within 90 seconds')
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


class _FakeStreamingDataLoader:
    """Mimics a streaming DataLoader: batches are produced in order and cannot be indexed."""

    def __init__(self, num_batches, batch_size):
        self.batches = [list(range(i * batch_size, (i + 1) * batch_size)) for i in range(num_batches)]

    def __iter__(self):
        return iter(self.batches)


class TestComputeStreamingSkipBatches(unittest.TestCase):

    def test_zero_when_nothing_consumed(self):
        self.assertEqual(compute_streaming_skip_batches(0, 6, 32), 0)

    def test_exact_multiple(self):
        # micro_batch_size=6, dp_world_size=32 -> 192 samples per dispatcher step
        self.assertEqual(compute_streaming_skip_batches(192, 6, 32), 1)
        self.assertEqual(compute_streaming_skip_batches(3 * 192, 6, 32), 3)

    def test_non_divisible_floors_towards_replay(self):
        # A remainder replays instead of skipping past unseen samples.
        self.assertEqual(compute_streaming_skip_batches(199, 6, 32), 1)
        self.assertEqual(compute_streaming_skip_batches(383, 6, 32), 1)

    def test_grpo_generated_samples_normalized_to_prompts(self):
        # Checkpoints count generated samples; the dataloader iterates
        # unrepeated prompts. 8 generated samples with num_generations=4 are
        # 2 prompts, i.e. 2 dispatcher batches at micro_batch_size=1, DP=1.
        self.assertEqual(compute_streaming_skip_batches(8, 1, 1, num_generations=4), 2)
        # Without the conversion the same checkpoint would skip 8 batches,
        # i.e. past 4x unseen prompts.
        self.assertEqual(compute_streaming_skip_batches(8, 1, 1), 8)

    def test_non_divisible_num_generations_raises(self):
        # Mirrors the non-streaming sampler contract: such a checkpoint is invalid.
        with self.assertRaises(RuntimeError):
            compute_streaming_skip_batches(9, 1, 1, num_generations=4)


class TestDispatcherSkipBatches(unittest.TestCase):
    """Single-process (world_size=1) check of the skip mechanics the formula feeds."""

    def test_skip_starts_after_consumed_batches(self):
        base = _FakeStreamingDataLoader(num_batches=6, batch_size=2)
        dispatcher = DataLoaderDispatcher(base, skip_batches=2)

        yielded = list(dispatcher)

        self.assertEqual(yielded[0], [4, 5])  # batches 0 and 1 were consumed pre-checkpoint
        self.assertEqual(len(yielded), 4)

    def test_skip_consumed_once_across_restarts(self):
        # cyclic_iter restarts the same dispatcher at the next epoch: the
        # resume offset applies to the first pass only, later epochs must be
        # read in full instead of skipping their heads again.
        base = _FakeStreamingDataLoader(num_batches=4, batch_size=2)
        dispatcher = DataLoaderDispatcher(base, skip_batches=2)

        first = list(dispatcher)
        second = list(dispatcher)

        self.assertEqual(first, [[4, 5], [6, 7]])
        self.assertEqual(second, [[0, 1], [2, 3], [4, 5], [6, 7]])

    def test_skip_offset_beyond_epoch_terminates_cleanly(self):
        # Only rank 0 pulls from the stream; an offset larger than the pass
        # must not raise out of the skip loop (which would leave the other
        # ranks waiting for their first scatter).
        base = _FakeStreamingDataLoader(num_batches=4, batch_size=2)
        dispatcher = DataLoaderDispatcher(base, skip_batches=10)

        self.assertEqual(list(dispatcher), [])
        # A later restart reads the full epoch.
        self.assertEqual(len(list(dispatcher)), 4)

    def test_no_skip_keeps_first_batch(self):
        base = _FakeStreamingDataLoader(num_batches=6, batch_size=2)
        dispatcher = DataLoaderDispatcher(base)

        self.assertEqual(next(iter(dispatcher)), [0, 1])


if __name__ == '__main__':
    unittest.main()

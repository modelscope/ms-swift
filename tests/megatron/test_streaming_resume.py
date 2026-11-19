# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest

from swift.dataloader.dispatcher import DataLoaderDispatcher, compute_streaming_skip_batches


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


class TestDispatcherSkipBatches(unittest.TestCase):
    """Single-process (world_size=1) check of the skip mechanics the formula feeds."""

    def test_skip_starts_after_consumed_batches(self):
        base = _FakeStreamingDataLoader(num_batches=6, batch_size=2)
        dispatcher = DataLoaderDispatcher(base, skip_batches=2)

        yielded = list(dispatcher)

        self.assertEqual(yielded[0], [4, 5])  # batches 0 and 1 were consumed pre-checkpoint
        self.assertEqual(len(yielded), 4)

    def test_no_skip_keeps_first_batch(self):
        base = _FakeStreamingDataLoader(num_batches=6, batch_size=2)
        dispatcher = DataLoaderDispatcher(base)

        self.assertEqual(next(iter(dispatcher)), [0, 1])


if __name__ == '__main__':
    unittest.main()

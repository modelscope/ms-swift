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

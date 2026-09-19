# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for host-side packed CP boundaries: what ``prepare_batch`` caches, and the
reconstruction consumer that reads them.

Written as ``unittest.TestCase`` so the repository CI collects them: the runner
uses ``unittest.defaultTestLoader.discover`` (``tests/run.py``), which ignores
bare pytest functions.
"""
import torch
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch


def _require_trainer_utils():
    """Import the Megatron-dependent module, or skip.

    Kept out of module scope so that a runner without the Megatron stack reports
    these tests as skipped instead of failing to import the whole module, which
    ``unittest`` discovery turns into an error.
    """
    try:
        import swift.megatron.trainers.utils as trainer_utils
    except Exception as exc:  # ImportError, or env init failure inside swift.megatron
        raise unittest.SkipTest(f'Megatron stack unavailable: {exc}')
    return trainer_utils


class _FakeMPU:

    def __init__(self, size, rank):
        self.size = size
        self.rank = rank

    def get_context_parallel_world_size(self):
        return self.size

    def get_context_parallel_rank(self):
        return self.rank

    def get_context_parallel_group(self):
        return None


def _build_args(cp_size=2):
    return SimpleNamespace(
        task_type='causal_lm',
        pipeline_model_parallel_size=1,
        padding_free=True,
        is_multimodal=False,
        context_parallel_size=cp_size,
        cp_partition_mode='zigzag',
    )


def _build_data(lengths, include_seq_lens=True):
    position_ids = torch.tensor([[index for length in lengths for index in range(length)]],
                                dtype=torch.long,
                                device='cuda')
    input_ids = torch.arange(sum(lengths), device='cuda', dtype=torch.long).view(1, -1)
    data = {
        'input_ids': input_ids,
        'labels': input_ids.clone(),
        'position_ids': position_ids,
    }
    if include_seq_lens:
        data['seq_lens'] = lengths
        data['num_samples'] = len(lengths)
    return data


def _patched_prepare_batch(args, data, cp_size, cp_rank):
    trainer_utils = _require_trainer_utils()
    fake_mpu = _FakeMPU(cp_size, cp_rank)
    with patch('swift.megatron.trainers.utils.mpu',
               fake_mpu), patch('swift.megatron.utils.megatron_lm_utils.mpu',
                                fake_mpu), patch('mcore_bridge.utils.megatron_utils.mpu', fake_mpu):
        return trainer_utils.prepare_batch(args, data)


@unittest.skipIf(not torch.cuda.is_available(), 'CUDA is required for packed CP metadata integration')
class TestPackedCpMetadata(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _require_trainer_utils()
        torch.cuda.set_device(0)

    def test_prepare_batch_caches_host_metadata_before_cp_split(self):
        lengths = [4, 8]
        batch = _patched_prepare_batch(_build_args(), _build_data(lengths), cp_size=2, cp_rank=0)

        packed = batch['packed_seq_params']
        self.assertEqual(packed.seq_lens.tolist(), lengths)
        self.assertEqual(packed.swift_cu_seqlens, (0, 4, 12))

    def test_prepare_batch_skips_boundaries_without_host_seq_lens(self):
        """Without host-side ``seq_lens`` the boundaries must be omitted, not rebuilt from CUDA."""
        batch = _patched_prepare_batch(_build_args(), _build_data([4, 8], include_seq_lens=False), cp_size=2, cp_rank=0)

        packed = batch['packed_seq_params']
        self.assertIsNotNone(packed)
        self.assertFalse(hasattr(packed, 'swift_cu_seqlens'), 'swift_cu_seqlens must be absent without host seq_lens')

    def test_prepare_batch_skips_boundaries_at_cp_one(self):
        """Only CP > 1 reconstructs, so CP=1 must not pay for boundaries nobody reads."""
        batch = _patched_prepare_batch(_build_args(cp_size=1), _build_data([4, 8]), cp_size=1, cp_rank=0)

        packed = batch['packed_seq_params']
        self.assertIsNotNone(packed)
        self.assertFalse(hasattr(packed, 'swift_cu_seqlens'), 'swift_cu_seqlens must be absent at CP=1')


def _boundaries(seq_lens, cp_size, pad_to=None):
    """Return the real sample boundaries and every boundary visible in ``cu_seqlens_q``.

    Production pads packed ``position_ids`` with ``arange(2 * cp_size)`` repeated, so
    every padding block of ``2 * cp_size`` tokens restarts at 0 and therefore looks
    like another packed sample to ``cu_seqlens_q``.
    """
    real_boundaries = [0]
    for seq_len in seq_lens:
        real_boundaries.append(real_boundaries[-1] + seq_len)
    real_total = real_boundaries[-1]

    all_boundaries = list(real_boundaries)
    if pad_to is not None:
        block = 2 * cp_size
        assert (pad_to - real_total) % block == 0, 'padding must be a whole number of blocks'
        for start in range(real_total, pad_to, block):
            all_boundaries.append(start + block)
    return real_boundaries, all_boundaries


def _zigzag_shards(seq_lens, cp_size, pad_to=None):
    """Build the CP-local shards a zigzag split produces, optionally with packed padding."""
    real_boundaries, all_boundaries = _boundaries(seq_lens, cp_size, pad_to)
    total = all_boundaries[-1]
    full = torch.arange(total, dtype=torch.float32).reshape(1, -1)
    shards = [torch.zeros(1, total // cp_size) for _ in range(cp_size)]
    for index in range(len(all_boundaries) - 1):
        start, end = all_boundaries[index], all_boundaries[index + 1]
        chunk = (end - start) // cp_size
        half = chunk // 2
        local_start = start // cp_size
        for rank in range(cp_size):
            shards[rank][0, local_start:local_start + half] = full[0, start + rank * half:start + (rank + 1) * half]
            shards[rank][0, local_start + half:local_start + chunk] = full[0, end - (rank + 1) * half:end - rank * half]
    return full, real_boundaries, all_boundaries, shards


def _contiguous_shards(seq_lens, cp_size, pad_to=None):
    """Contiguous CP splits the whole flattened packed sequence across ranks."""
    real_boundaries, all_boundaries = _boundaries(seq_lens, cp_size, pad_to)
    total = all_boundaries[-1]
    assert total % cp_size == 0, 'contiguous CP requires the packed length to divide by cp_size'
    full = torch.arange(total, dtype=torch.float32).reshape(1, -1)
    return full, real_boundaries, all_boundaries, list(torch.chunk(full, cp_size, dim=1))


def _reconstruct(seq_lens, cp_size, use_host_metadata, forbid_item=False, pad_to=None, mode='zigzag'):
    trainer_utils = _require_trainer_utils()
    builder = _zigzag_shards if mode == 'zigzag' else _contiguous_shards
    full, real_boundaries, all_boundaries, shards = builder(seq_lens, cp_size, pad_to)
    packed_seq_params = SimpleNamespace(cu_seqlens_q=torch.tensor(all_boundaries, dtype=torch.int32))
    if use_host_metadata:
        packed_seq_params.swift_cu_seqlens = tuple(real_boundaries)

    def fake_all_gather(output_list, tensor, group=None):
        for index, shard in enumerate(shards):
            output_list[index].copy_(shard)

    sync_guard = patch.object(
        torch.Tensor, 'item',
        side_effect=AssertionError('host metadata path must not synchronize')) if forbid_item else nullcontext()
    with patch.object(trainer_utils, 'mpu', _FakeMPU(cp_size, 0)), \
            patch('torch.distributed.all_gather', side_effect=fake_all_gather), sync_guard:
        result = trainer_utils.reconstruct_tensor_cp(cp_size, shards[0], packed_seq_params, len(seq_lens), mode)
    return result, full[:, :real_boundaries[-1]]


class TestPackedCpMetadataConsumer(unittest.TestCase):
    """``reconstruct_tensor_cp`` is the GRPO/RLHF consumer of the cached boundaries."""

    @classmethod
    def setUpClass(cls):
        _require_trainer_utils()

    CASES = (([8, 24], 2), ([16, 48], 4), ([4, 8, 12, 16], 2), ([8, 8, 8], 4))

    # Packed padding makes `cu_seqlens_q` longer than the cached boundaries, because
    # each padding block restarts `position_ids` at 0. Only the first
    # `num_samples + 1` entries are read, so the cached tuple must stay a valid
    # prefix -- this is the shape of a real padded training batch.
    PADDED_CASES = (([8, 24], 2, 40), ([16, 48], 4, 96), ([4, 8, 12], 2, 40))

    def test_host_metadata_matches_cuda_boundary_fallback(self):
        for seq_lens, cp_size in self.CASES:
            with self.subTest(seq_lens=seq_lens, cp_size=cp_size):
                fallback, full = _reconstruct(seq_lens, cp_size, use_host_metadata=False)
                host, _ = _reconstruct(seq_lens, cp_size, use_host_metadata=True)
                torch.testing.assert_close(fallback, full)
                torch.testing.assert_close(host, full)
                torch.testing.assert_close(host, fallback)

    def test_host_metadata_matches_fallback_with_packed_padding(self):
        for seq_lens, cp_size, pad_to in self.PADDED_CASES:
            with self.subTest(seq_lens=seq_lens, cp_size=cp_size, pad_to=pad_to):
                _, real_boundaries, all_boundaries, _ = _zigzag_shards(seq_lens, cp_size, pad_to)
                # Guard the premise: padding must actually lengthen cu_seqlens_q,
                # otherwise this case would silently degenerate into the unpadded one.
                self.assertGreater(len(all_boundaries), len(real_boundaries))

                fallback, full = _reconstruct(seq_lens, cp_size, use_host_metadata=False, pad_to=pad_to)
                host, _ = _reconstruct(seq_lens, cp_size, use_host_metadata=True, pad_to=pad_to)
                torch.testing.assert_close(fallback, full)
                torch.testing.assert_close(host, full)
                torch.testing.assert_close(host, fallback)

    def test_host_metadata_path_issues_no_device_sync(self):
        for pad_to in (None, 40):
            with self.subTest(pad_to=pad_to):
                result, full = _reconstruct([8, 24], 2, use_host_metadata=True, forbid_item=True, pad_to=pad_to)
                torch.testing.assert_close(result, full)

    def test_falls_back_when_host_metadata_is_absent(self):
        # PackedSeqParams built before this change carry no Swift boundaries, so the
        # original CUDA path must keep working.
        result, full = _reconstruct([8, 24], 2, use_host_metadata=False)
        torch.testing.assert_close(result, full)

    def test_contiguous_host_metadata_matches_cuda_boundary_fallback(self):
        """The contiguous branch reads the same cached boundary to size its truncation.

        `contiguous` is not production-supported, but this PR touches its truncation
        line, so the line needs coverage.
        """
        for seq_lens, cp_size, pad_to in (([8, 24], 2, None), ([8, 24], 2, 40), ([16, 48], 4, 96)):
            with self.subTest(seq_lens=seq_lens, cp_size=cp_size, pad_to=pad_to):
                fallback, full = _reconstruct(
                    seq_lens, cp_size, use_host_metadata=False, pad_to=pad_to, mode='contiguous')
                host, _ = _reconstruct(seq_lens, cp_size, use_host_metadata=True, pad_to=pad_to, mode='contiguous')
                torch.testing.assert_close(fallback, full)
                torch.testing.assert_close(host, full)
                torch.testing.assert_close(host, fallback)

    def test_contiguous_host_metadata_path_issues_no_device_sync(self):
        result, full = _reconstruct([8, 24], 2, use_host_metadata=True, forbid_item=True, pad_to=40, mode='contiguous')
        torch.testing.assert_close(result, full)


if __name__ == '__main__':
    unittest.main()

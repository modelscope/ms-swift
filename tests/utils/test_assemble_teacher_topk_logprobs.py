"""Unit tests for ``assemble_teacher_topk_logprobs``.

Covers padding_free (packed) and non-packed modes.
"""
import pytest
import torch

from swift.rlhf_trainers.utils import assemble_teacher_topk_logprobs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parsed(seq_len: int, topk: int):
    """Create ``parsed`` data mimicking ``parse_prompt_logprobs`` output.

    ``parse_prompt_logprobs`` skips position 0, so ``len(lps) == seq_len - 1``.
    Each position gets ``topk`` values; we use distinct floats so we can verify
    the mapping.
    """
    lps = []
    ixs = []
    for pos in range(seq_len - 1):  # skip position 0
        lps.append([float(pos * 100 + k) for k in range(topk)])
        ixs.append([pos * 10 + k for k in range(topk)])
    return lps, ixs


# ---------------------------------------------------------------------------
# 1. padding_free=True (packed mode)
# ---------------------------------------------------------------------------


class TestPacked:

    def test_single_sample(self):
        """One sample, sequential mapping."""
        topk = 2
        seq_len = 6
        parsed = [_make_parsed(seq_len, topk)]
        cu_seqlens = [0, seq_len]

        out_lp, out_ix = assemble_teacher_topk_logprobs(
            parsed, batch_size=1, seq_len=seq_len, cu_seqlens=cu_seqlens, topk=topk, device=torch.device('cpu'))

        assert out_lp.shape == (1, seq_len, topk)
        # Positions 0..4 filled, position 5 = -inf
        for i in range(seq_len - 1):
            for k in range(topk):
                assert out_lp[0, i, k].item() == pytest.approx(float(i * 100 + k))
                assert out_ix[0, i, k].item() == i * 10 + k
        assert torch.isinf(out_lp[0, seq_len - 1, 0])

    def test_two_samples(self):
        """Two samples packed together."""
        topk = 1
        s1, s2 = 4, 5
        parsed = [_make_parsed(s1, topk), _make_parsed(s2, topk)]
        cu_seqlens = [0, s1, s1 + s2]

        out_lp, out_ix = assemble_teacher_topk_logprobs(
            parsed, batch_size=1, seq_len=s1 + s2, cu_seqlens=cu_seqlens, topk=topk, device=torch.device('cpu'))

        assert out_lp.shape == (1, s1 + s2, topk)
        # Sample 1: positions 0..2 filled, position 3 = -inf
        assert out_lp[0, 0, 0].item() == pytest.approx(0.0)
        assert out_lp[0, 2, 0].item() == pytest.approx(200.0)
        assert torch.isinf(out_lp[0, 3, 0])
        # Sample 2: positions 4..7 filled, position 8 = -inf
        assert out_lp[0, 4, 0].item() == pytest.approx(0.0)
        assert out_lp[0, 7, 0].item() == pytest.approx(300.0)
        assert torch.isinf(out_lp[0, 8, 0])


# ---------------------------------------------------------------------------
# 2. padding_free=False (non-packed mode)
# ---------------------------------------------------------------------------


class TestNonPacked:

    def test_no_offset(self):
        """Batch of 2, no left padding (offsets=0)."""
        topk = 2
        seq_len = 5
        batch_size = 2
        parsed = [_make_parsed(seq_len, topk), _make_parsed(seq_len, topk)]

        out_lp, out_ix = assemble_teacher_topk_logprobs(
            parsed, batch_size=batch_size, seq_len=seq_len, cu_seqlens=None, topk=topk, device=torch.device('cpu'))

        assert out_lp.shape == (batch_size, seq_len, topk)
        for b in range(batch_size):
            lps = parsed[b][0]
            for i in range(seq_len - 1):
                for k in range(topk):
                    assert out_lp[b, i, k].item() == pytest.approx(lps[i][k])
            assert torch.isinf(out_lp[b, seq_len - 1, 0])

    def test_with_offsets(self):
        """Batch of 2 with left padding (offsets=[2, 0])."""
        topk = 1
        seq_len = 6
        batch_size = 2
        parsed = [_make_parsed(4, topk), _make_parsed(6, topk)]
        offsets = [2, 0]

        out_lp, out_ix = assemble_teacher_topk_logprobs(
            parsed,
            batch_size=batch_size,
            seq_len=seq_len,
            cu_seqlens=None,
            topk=topk,
            device=torch.device('cpu'),
            offsets=offsets)

        assert out_lp.shape == (batch_size, seq_len, topk)
        # Sample 0: starts at offset 2, has 3 logprobs (4 tokens - 1)
        lps0 = parsed[0][0]
        assert out_lp[0, 2, 0].item() == pytest.approx(lps0[0][0])
        assert out_lp[0, 4, 0].item() == pytest.approx(lps0[2][0])
        assert torch.isinf(out_lp[0, 5, 0])  # last position for sample 0
        assert torch.isinf(out_lp[0, 0, 0])  # left padding
        assert torch.isinf(out_lp[0, 1, 0])  # left padding

        # Sample 1: starts at offset 0, has 5 logprobs (6 tokens - 1)
        lps1 = parsed[1][0]
        assert out_lp[1, 0, 0].item() == pytest.approx(lps1[0][0])
        assert out_lp[1, 4, 0].item() == pytest.approx(lps1[4][0])
        assert torch.isinf(out_lp[1, 5, 0])

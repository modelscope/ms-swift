"""S8c — RewardWorker algorithmic half: PRM fusion, sequence KL, teacher-signal reweighting (L1).

``RewardWorker`` (@remote_class) is the dev reward actor; under the default local mode it can be
constructed and called directly (``RewardWorker(orm_scorers=[])``), so the numeric kernels are pinned
with hand-computed oracles and no Ray/GPU. ``reweight_advantages`` is the regression anchor for the
[B,1]→[B] base-advantage fix (the twinkle estimators broadcast [B] themselves; a [B,1] base made
``unsqueeze(1)`` a [B,1,1] tensor that cannot ``expand_as`` the [B,T] mask -- any RLSD/SDAR GRPO run
crashed at run_grpo's reweight call).

Maps to ut_plan S8-56 (fuse_prm convex blend) plus the sequence_kl / reweight kernels behind it.
"""
import math

import pytest


@pytest.fixture
def worker():
    from swift.dev.rewards.worker import RewardWorker

    return RewardWorker(orm_scorers=[])


def _assert_rows(result, expected):
    """Row-wise approx: pytest.approx does not support nested list-of-list."""
    assert len(result) == len(expected)
    for row, want in zip(result, expected):
        assert row == pytest.approx(want)


class TestFusePRM:

    def test_convex_blend(self, worker):
        """S8-56: per-token advantage = mix*process + (1-mix)*orm -- a genuine convex blend per token."""
        out = worker.fuse_prm([1.0, 2.0], [[0.5, 0.5], [0.0, 1.0]], 0.5)
        _assert_rows(out, [[0.75, 0.75], [1.0, 1.5]])
        # mix endpoints: pure ORM at 0, pure process at 1.
        _assert_rows(worker.fuse_prm([1.0], [[2.0, 4.0]], 0.0), [[1.0, 1.0]])
        _assert_rows(worker.fuse_prm([1.0], [[2.0, 4.0]], 1.0), [[2.0, 4.0]])

    def test_missing_process_rewards_fail_loudly(self, worker):
        """S8-56 guard: a PRM channel configured (mix set) but producing no per-sample process reward
        must raise, not silently fall back to the ORM advantage (the channel would be a silent no-op)."""
        with pytest.raises(RuntimeError, match='per-sample process reward'):
            worker.fuse_prm([1.0], [None], 0.5)


class TestSequenceKL:

    def test_k3_estimator(self, worker):
        """Sequence KL = mean(exp(d) - d - 1) with d = ref - old: d=1 for both tokens -> e - 2."""
        out = worker.sequence_kl([[-1.0, -1.0]], [[0.0, 0.0]])
        assert out == pytest.approx([math.e - 2.0])

    def test_identical_logps_are_zero_kl(self, worker):
        out = worker.sequence_kl([[-0.3, 0.7, 1.2]], [[-0.3, 0.7, 1.2]])
        assert out == pytest.approx([0.0], abs=1e-6)

    def test_misaligned_rows_rejected(self, worker):
        """Guard: old/ref token counts must match per sample -- a silent elementwise op would broadcast
        or crash deep inside torch instead of naming the offending pair."""
        with pytest.raises(RuntimeError, match='reference logps misaligned'):
            worker.sequence_kl([[-1.0, -1.0]], [[0.0]])


class TestReweightAdvantages:
    """The teacher-signal reweighting kernels (RLSD / SDAR / expand) shared by GRPO recipes."""

    def test_expand_broadcasts_scalar_per_token(self, worker):
        """REGRESSION ANCHOR: the base advantage reaches the twinkle estimator as [B] (1-D); the
        estimator broadcasts it per token. A [B,1] base crashed every expand/RLSD call with
        ``expand(torch.FloatTensor{[B, 1, 1]}, size=[B, T])`` -- re-pin it so the fix cannot regress."""
        out = worker.reweight_advantages([2.0, -1.0], [[-0.5, -0.5], [-0.5, 0.0]],
                                         [[-0.5, -0.5], [-0.5, 0.0]], mode='expand', teacher_kl_coef=0.0)
        _assert_rows(out, [[2.0, 2.0], [-1.0, -1.0]])

    def test_rlsd_lam_zero_is_plain_broadcast(self, worker):
        """RLSD lam=0: the reweight factor is (1-0)+0*clip(...) = 1, so the base advantage broadcasts."""
        out = worker.reweight_advantages([2.0], [[-1.0, -1.0]], [[0.0, 0.0]], mode='rlsd', lam=0.0)
        _assert_rows(out, [[2.0, 2.0]])

    def test_rlsd_formula(self, worker):
        """RLSD lam=1, delta = teacher-old = 1, A = 1: w = exp(sign(A)*delta) = e, A_hat = A*w = e."""
        out = worker.reweight_advantages([1.0], [[-1.0, -1.0]], [[0.0, 0.0]], mode='rlsd', lam=1.0)
        _assert_rows(out, [[math.e, math.e]])

    def test_rlsd_clip_bounds_the_factor(self, worker):
        """clip_range=0.5 pins w to [1-0.5, 1+0.5]: A=1, delta=1 -> A_hat = 1*1.5."""
        out = worker.reweight_advantages([1.0], [[-1.0, -1.0]], [[0.0, 0.0]], mode='rlsd',
                                         lam=1.0, clip_range=0.5)
        _assert_rows(out, [[1.5, 1.5]])

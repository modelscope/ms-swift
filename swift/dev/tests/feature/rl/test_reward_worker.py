"""S8c — dev2 RewardWorker GRPO training-half kernels (L1, hand-computed).

``RewardWorker`` (swift/dev/rewards/worker.py) subclasses twinkle's ScoreWorker and keeps only GRPO's
training-half algorithms: the PRM×ORM convex fusion, the per-token teacher-signal reweight (RLSD/SDAR),
and the sequence-KL penalty. The scoring orchestration it INHERITS (compute-rewards-per-func / the
rule+RM column merge / score_group_prm) is twinkle-layer, covered by twinkle_agentic's own suite, so it
is not duplicated here. These three are dev2's own numerics; ``@remote_function`` defaults to local
mode, so they are directly callable without Ray.

Writing these found a real bug: reweight_advantages built the base advantage as [B, 1] while the twinkle
estimators take [B] and broadcast it themselves, so unsqueeze(1) made a [B, 1, 1] tensor that could not
expand_as the [B, T] mask -- any live RLSD/SDAR GRPO run would crash. Fixed to [B]; pinned below.

Maps to ut_plan S8-56 (PRM fusion) and the teacher-signal / KL kernels underpinning S7's RLSD + ref-KL e2e.
"""
import math

import pytest


@pytest.fixture
def worker():
    from swift.dev.rewards.worker import RewardWorker

    return RewardWorker(orm_scorers=[])


def _assert_rows(result, expected):
    """Row-wise approx compare (pytest.approx rejects nested list-of-lists)."""
    assert len(result) == len(expected)
    for got, want in zip(result, expected):
        assert got == pytest.approx(want)


# ----------------------------------------------------------------------
# S8-56 — fuse_prm: per-token convex blend of the process reward with the scalar ORM advantage
# ----------------------------------------------------------------------
def test_fuse_prm_convex_blend(worker):
    """S8-56: advantage[i] = mix*process[i] + (1-mix)*orm on every response token; mix=0 is pure GRPO,
    mix=1 pure PRM."""
    orm = [1.0, 2.0]
    process = [[0.5, 0.5], [0.0, 1.0]]
    # mix=0.5: sample0 [0.25+0.5, 0.25+0.5]=[0.75,0.75]; sample1 [0+1.0, 0.5+1.0]=[1.0,1.5]
    _assert_rows(worker.fuse_prm(orm, process, 0.5), [[0.75, 0.75], [1.0, 1.5]])
    _assert_rows(worker.fuse_prm(orm, process, 0.0), [[1.0, 1.0], [2.0, 2.0]])  # pure ORM broadcast
    _assert_rows(worker.fuse_prm(orm, process, 1.0), [[0.5, 0.5], [0.0, 1.0]])  # pure PRM


def test_fuse_prm_requires_process_reward(worker):
    """S8-56 guard: fusion needs a per-sample process reward (score_group_prm must run first)."""
    with pytest.raises(RuntimeError, match='requires a per-sample process reward'):
        worker.fuse_prm([1.0], [None], 0.5)


# ----------------------------------------------------------------------
# sequence_kl: the k3 estimator mean(exp(d) - d - 1), d = ref - old
# ----------------------------------------------------------------------
def test_sequence_kl_k3_estimator(worker):
    """sequence_kl is the non-negative k3 KL estimator mean(exp(d)-d-1) over each sample's tokens."""
    assert worker.sequence_kl([[0.0, 0.0]], [[0.0, 0.0]]) == pytest.approx([0.0])  # d=0 -> 0
    # d=1 on both tokens -> exp(1)-1-1 = e-2 each -> mean e-2
    assert worker.sequence_kl([[-1.0, -1.0]], [[0.0, 0.0]]) == pytest.approx([math.e - 2.0])
    # per-sample: two samples scored independently
    out = worker.sequence_kl([[0.0, 0.0], [-1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]])
    assert out == pytest.approx([0.0, math.e - 2.0])


def test_sequence_kl_rejects_misaligned_logps(worker):
    """sequence_kl guard: old and ref logps must be token-aligned per sample."""
    with pytest.raises(RuntimeError, match='reference logps misaligned'):
        worker.sequence_kl([[0.0, 0.0]], [[0.0]])


# ----------------------------------------------------------------------
# reweight_advantages: the per-token teacher-signal reweight (the [B]-shape bug regression anchor)
# ----------------------------------------------------------------------
def test_reweight_advantages_expand_broadcasts_scalar(worker):
    """expand mode with teacher_kl_coef=0 is a plain broadcast of the per-sequence advantage to every
    response token (the [B] base advantage expanded to [B, T] -- the path the shape bug crashed)."""
    out = worker.reweight_advantages(
        [2.0, -1.0], [[-0.5, -0.5], [-0.5, 0.0]], [[-0.5, -0.5], [-0.5, 0.0]],
        mode='expand', teacher_kl_coef=0.0)
    _assert_rows(out, [[2.0, 2.0], [-1.0, -1.0]])


def test_reweight_advantages_rlsd_formula(worker):
    """rlsd mode: lam=0 is a plain broadcast; lam=1 applies A_hat = A * clip(exp(sign(A)*delta), ...) with
    delta = teacher_logp - student_logp (teacher=1, student=0 -> exp(1)=e on a positive advantage)."""
    # lam=0 -> pure GRPO broadcast (no reweight)
    _assert_rows(
        worker.reweight_advantages(
            [2.0], [[-0.5, -0.5]], [[-0.5, -0.5]], mode='rlsd', lam=0.0, clip_range=5.0), [[2.0, 2.0]])
    # lam=1, teacher-student gap of 1 on a +1 advantage -> reweight = exp(1) = e -> A_hat = e
    out = worker.reweight_advantages(
        [1.0], [[0.0, 0.0]], [[1.0, 1.0]], mode='rlsd', lam=1.0, clip_range=5.0)
    _assert_rows(out, [[math.e, math.e]])

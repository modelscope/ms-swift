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
import types

import pytest
import torch


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


def _gae_inputs(lengths, *, seq_len, critic_values, old_logps, ref_logps, rm_scores, ref_present=True):
    """Script a PPO batch for ``RewardWorker.gae`` (no GPU / no Ray -- the worker is local-constructible).

    Each completion i carries ``lengths[i]`` response tokens laid out CONTIGUOUSLY at the END of a
    ``seq_len``-token sequence, so ``response_positions`` == ``[seq_len-L_i .. seq_len-1]``. The critic-value
    and reference frames are full-sequence ``[batch, seq_len]`` tensors -- exactly the shape a real
    ``forward_only`` returns -- which ``response_token_scores`` indexes at those positions to recover the
    per-response-token rows. ``ref_present=False`` drops the reference 'logps' so ``allow_missing`` fires
    (PPO reads a missing reference as "no KL penalty"). Samples are ``SimpleNamespace`` carrying only the
    three attributes ``gae`` reads (``encoded`` / ``orm_reward`` / ``rollout_logps``).
    """
    samples, values_rows, ref_rows = [], [], []
    for i, length in enumerate(lengths):
        labels = [-100] * (seq_len - length) + [1] * length
        samples.append(
            types.SimpleNamespace(
                encoded={'input_ids': [1] * seq_len, 'labels': labels},
                orm_reward=[rm_scores[i]],
                rollout_logps=list(old_logps[i])))
        vrow = [0.0] * seq_len
        if length:
            vrow[seq_len - length:] = list(critic_values[i])
        values_rows.append(vrow)
        rrow = [0.0] * seq_len
        if length:
            rrow[seq_len - length:] = list(ref_logps[i])
        ref_rows.append(rrow)
    scored = {'values': {'logits': torch.tensor(values_rows, dtype=torch.float32)}}
    scored['ref'] = ({'logps': torch.tensor(ref_rows, dtype=torch.float32)} if ref_present else {'logps': None})
    return samples, scored


def _gae_configs(*, kl_coef, gamma, lam, whiten):
    reward_config = types.SimpleNamespace(orm_weights=None)
    rl_config = types.SimpleNamespace(kl_coef=kl_coef, gamma=gamma, lam=lam, whiten_rewards=whiten)
    return reward_config, rl_config


def _rectangular_oracle(lengths, rm_scores, old_logps, ref_logps, critic_values, *, gamma, lam, kl_coef,
                        normalize):
    """Independent oracle for ``gae``'s pad/mask/unpad glue: build the SAME rectangular [batch, max_len]
    rewards/values + valid-token mask BY HAND (not via gae's padding code), run the real ``GAEAdvantage``,
    then unpad by hand. ``gae`` must reproduce this exactly -- if its padding/masking drifts (or it feeds
    ragged rows straight to ``torch.as_tensor``, the original crash) the two diverge or gae raises."""
    from twinkle.advantage import GAEAdvantage

    token_rewards = GAEAdvantage.build_token_rewards(
        rm_scores, lengths, old_logps=old_logps, ref_logps=ref_logps, kl_coef=kl_coef)
    max_len = max(lengths)
    rewards_pad = [list(row) + [0.0] * (max_len - len(row)) for row in token_rewards]
    values_pad = [list(row) + [0.0] * (max_len - len(row)) for row in critic_values]
    masks = [[True] * length + [False] * (max_len - length) for length in lengths]
    adv, ret = GAEAdvantage(gamma=gamma, gae_lambda=lam, normalize=normalize)(rewards_pad, values_pad,
                                                                              masks=masks)
    return ([adv[i, :length].tolist() for i, length in enumerate(lengths)],
            [ret[i, :length].tolist() for i, length in enumerate(lengths)])


class TestGAE:
    """PPO's GAE anchor builder. REGRESSION ANCHOR for the ragged-input crash: a batch's completions stop at
    different lengths, so per-completion rewards/values are ragged, but ``GAEAdvantage`` consumes a
    rectangular ``[batch, max_len]`` tensor + a valid-token mask. Feeding ragged rows straight in raised
    ``ValueError: expected sequence of length N at dim 1 (got M)`` at the FIRST PPO step (the path had no
    test, so the bug shipped latent). ``gae`` must pad -> mask -> run GAE -> unpad back to each completion's
    own length, token-aligned with the values/old_logps that ``_train_ppo_batch`` replays against."""

    def test_ragged_batch_matches_rectangular_oracle(self, worker):
        """#1 THE BUG: lengths [2,5,3] (uneven) with a live KL penalty. Before the fix this raised at
        torch.as_tensor; now advantages/returns come back ragged [2,5,3] and equal the hand-padded oracle."""
        lengths = [2, 5, 3]
        critic_values = [[0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7]]
        old_logps = [[-0.5, -0.6], [-0.1, -0.2, -0.3, -0.4, -0.5], [-0.7, -0.8, -0.9]]
        ref_logps = [[-0.4, -0.5], [-0.15, -0.25, -0.35, -0.45, -0.55], [-0.6, -0.7, -0.8]]
        rm_scores = [1.0, 0.5, -0.5]
        samples, scored = _gae_inputs(
            lengths, seq_len=8, critic_values=critic_values, old_logps=old_logps, ref_logps=ref_logps,
            rm_scores=rm_scores)
        reward_config, rl_config = _gae_configs(kl_coef=0.05, gamma=1.0, lam=0.95, whiten=True)

        kept, advantages, returns, values, old = worker.gae(samples, scored, reward_config, rl_config)

        assert kept == [0, 1, 2]
        assert [len(row) for row in advantages] == lengths
        assert [len(row) for row in returns] == lengths
        assert [len(row) for row in values] == lengths
        _assert_rows(old, old_logps)
        exp_adv, exp_ret = _rectangular_oracle(
            lengths, rm_scores, old_logps, ref_logps, critic_values, gamma=1.0, lam=0.95, kl_coef=0.05,
            normalize=True)
        _assert_rows(advantages, exp_adv)
        _assert_rows(returns, exp_ret)

    def test_single_completion_hand_computed(self, worker):
        """#2 FULLY-INDEPENDENT anchor (no twinkle in the expected value): one completion L=2, kl_coef=0 so
        token_rewards=[0, rm]=[0,1], gamma=1, lam=0.95, v=[0.3,0.4], normalize off. Hand GAE recurrence:
        A1 = r1 + gamma*0 - v1 = 1 - 0.4 = 0.6; A0 = (r0 + gamma*v1 - v0) + gamma*lam*A1
        = (0 + 0.4 - 0.3) + 0.95*0.6 = 0.67; returns = A + v -> [0.97, 1.0]."""
        samples, scored = _gae_inputs(
            [2], seq_len=4, critic_values=[[0.3, 0.4]], old_logps=[[-0.5, -0.5]],
            ref_logps=[[-0.5, -0.5]], rm_scores=[1.0])
        reward_config, rl_config = _gae_configs(kl_coef=0.0, gamma=1.0, lam=0.95, whiten=False)

        kept, advantages, returns, values, old = worker.gae(samples, scored, reward_config, rl_config)

        assert kept == [0]
        _assert_rows(advantages, [[0.67, 0.6]])
        _assert_rows(returns, [[0.97, 1.0]])
        _assert_rows(values, [[0.3, 0.4]])

    def test_equal_lengths_padding_is_noop(self, worker):
        """#3 already-rectangular batch (all L=3): the pad/mask path must still run and equal the oracle
        (max_len == every length, so the mask is all-True and padding adds nothing)."""
        lengths = [3, 3, 3]
        critic_values = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        old_logps = [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]]
        ref_logps = [[-0.2, -0.3, -0.4], [-0.5, -0.6, -0.7], [-0.8, -0.9, -1.0]]
        rm_scores = [1.0, 0.0, -1.0]
        samples, scored = _gae_inputs(
            lengths, seq_len=5, critic_values=critic_values, old_logps=old_logps, ref_logps=ref_logps,
            rm_scores=rm_scores)
        reward_config, rl_config = _gae_configs(kl_coef=0.1, gamma=0.99, lam=0.9, whiten=True)

        kept, advantages, returns, values, old = worker.gae(samples, scored, reward_config, rl_config)

        assert [len(row) for row in advantages] == lengths
        exp_adv, exp_ret = _rectangular_oracle(
            lengths, rm_scores, old_logps, ref_logps, critic_values, gamma=0.99, lam=0.9, kl_coef=0.1,
            normalize=True)
        _assert_rows(advantages, exp_adv)
        _assert_rows(returns, exp_ret)

    def test_critic_value_length_mismatch_raises(self, worker):
        """#4 GUARD: the critic values are recovered at the ENCODED response positions (3 tokens), but the
        sampler's rollout_logps carry only 2 -- a token misalignment between the rollout and the encoding.
        With the reference absent (ref_rows = old_logps, so build_token_rewards sees consistent 2-length
        rows) the mismatch surfaces at gae's own guard, which must name it rather than pad a wrong-length
        value into the GAE."""
        samples, scored = _gae_inputs(
            [3], seq_len=5, critic_values=[[0.1, 0.2, 0.3]], old_logps=[[-0.1, -0.2]],
            ref_logps=[[0.0, 0.0, 0.0]], rm_scores=[1.0], ref_present=False)
        reward_config, rl_config = _gae_configs(kl_coef=0.0, gamma=1.0, lam=0.95, whiten=False)

        with pytest.raises(ValueError, match='critic value length does not match'):
            worker.gae(samples, scored, reward_config, rl_config)

    def test_missing_reference_zeroes_the_kl_penalty(self, worker):
        """#5 no-KL branch: a reference frame with no 'logps' (allow_missing) makes gae reuse the old logps
        as the reference, so every per-token KL term -kl_coef*(old-ref) is 0 even with kl_coef>0 -- the
        result must equal the kl_coef=0 oracle (terminal reward only)."""
        lengths = [2, 4]
        critic_values = [[0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]
        old_logps = [[-0.5, -0.6], [-0.1, -0.2, -0.3, -0.4]]
        rm_scores = [1.0, 0.5]
        samples, scored = _gae_inputs(
            lengths, seq_len=6, critic_values=critic_values, old_logps=old_logps,
            ref_logps=[[0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], rm_scores=rm_scores, ref_present=False)
        reward_config, rl_config = _gae_configs(kl_coef=0.5, gamma=1.0, lam=0.95, whiten=False)

        kept, advantages, returns, values, old = worker.gae(samples, scored, reward_config, rl_config)

        # ref absent -> ref_rows = old_logps -> KL term 0 -> identical to kl_coef=0 with the same old_logps
        exp_adv, exp_ret = _rectangular_oracle(
            lengths, rm_scores, old_logps, old_logps, critic_values, gamma=1.0, lam=0.95, kl_coef=0.0,
            normalize=False)
        _assert_rows(advantages, exp_adv)
        _assert_rows(returns, exp_ret)

    def test_normalize_off_leaves_advantages_unwhitened(self, worker):
        """#6 whiten_rewards=False must thread normalize=False into GAEAdvantage: the advantages are the raw
        GAE (not zero-mean/unit-std across the batch). Pinned against the normalize=False oracle."""
        lengths = [2, 3]
        critic_values = [[0.3, 0.4], [0.1, 0.2, 0.3]]
        old_logps = [[-0.5, -0.6], [-0.1, -0.2, -0.3]]
        ref_logps = [[-0.4, -0.5], [-0.15, -0.25, -0.35]]
        rm_scores = [2.0, -1.0]
        samples, scored = _gae_inputs(
            lengths, seq_len=5, critic_values=critic_values, old_logps=old_logps, ref_logps=ref_logps,
            rm_scores=rm_scores)
        reward_config, rl_config = _gae_configs(kl_coef=0.05, gamma=1.0, lam=0.95, whiten=False)

        kept, advantages, returns, values, old = worker.gae(samples, scored, reward_config, rl_config)

        exp_adv, exp_ret = _rectangular_oracle(
            lengths, rm_scores, old_logps, ref_logps, critic_values, gamma=1.0, lam=0.95, kl_coef=0.05,
            normalize=False)
        _assert_rows(advantages, exp_adv)
        _assert_rows(returns, exp_ret)
        # the unwhitened advantages are NOT zero-mean across the batch (a whitening leak would make them so)
        flat = [a for row in advantages for a in row]
        assert abs(sum(flat) / len(flat)) > 1e-6

    def test_sample_without_response_tokens_is_dropped(self, worker):
        """#7 kept filter: a completion whose labels are all -100 has no trainable response token, so it is
        dropped from `kept` and the ragged outputs re-index to the surviving completions only."""
        lengths = [2, 0, 3]
        critic_values = [[0.3, 0.4], [], [0.1, 0.2, 0.3]]
        old_logps = [[-0.5, -0.6], [], [-0.1, -0.2, -0.3]]
        ref_logps = [[-0.4, -0.5], [], [-0.15, -0.25, -0.35]]
        rm_scores = [1.0, 9.9, 0.5]  # the dropped sample's reward must not leak into the kept rows
        samples, scored = _gae_inputs(
            lengths, seq_len=5, critic_values=critic_values, old_logps=old_logps, ref_logps=ref_logps,
            rm_scores=rm_scores)
        reward_config, rl_config = _gae_configs(kl_coef=0.05, gamma=1.0, lam=0.95, whiten=True)

        kept, advantages, returns, values, old = worker.gae(samples, scored, reward_config, rl_config)

        assert kept == [0, 2]
        assert [len(row) for row in advantages] == [2, 3]
        kept_lengths = [2, 3]
        exp_adv, exp_ret = _rectangular_oracle(
            kept_lengths, [rm_scores[0], rm_scores[2]], [old_logps[0], old_logps[2]],
            [ref_logps[0], ref_logps[2]], [critic_values[0], critic_values[2]], gamma=1.0, lam=0.95,
            kl_coef=0.05, normalize=True)
        _assert_rows(advantages, exp_adv)
        _assert_rows(returns, exp_ret)

"""S8a — GRPO advanced importance-sampling kernels (L1).

Two IS surfaces on the twinkle GRPOLoss the dev config drives:
``rollout_importance_sampling_mode`` (the policy-vs-SAMPLER gap correction, four modes + off) and
``importance_sampling_level`` (the policy-vs-OLD-POLICY ratio the PPO term uses, three levels --
token / sequence (GSPO) / sequence_token (GSPO-token gradient trick)). Each mode/level is pinned
with a hand-computed oracle on two tokens with ratios [0.5, 3.0].

CHORD's dual-RPC serialization and the mu warmup/cosine schedule are inline in run_grpo's training
loop (no pure function to pin) -- they are covered by the S7 chord e2e instead (S7-43).

Maps to ut_plan S8-09 (alignment-adjacent IS levels) and S8-14 (rollout IS modes).
"""
import math

import pytest
import torch


class _StubModel:

    def __init__(self):
        self.losses = {}

    def set_loss(self, loss, loss_key=None):
        self.losses[loss_key] = loss


def _grpo_loss(**overrides):
    from swift.dev.config import RLConfig
    from swift.dev.loss import configure_rl_loss

    model = _StubModel()
    configure_rl_loss(model, RLConfig(rl_type='grpo', **overrides))
    return model.losses.get(None) or next(iter(model.losses.values()))


def test_rollout_is_weights_off_is_raw_ratio():
    """Mode None: the truncated ratio itself (log clamped +-20), no correction."""
    loss = _grpo_loss()
    log_ratio = torch.log(torch.tensor([0.5, 3.0]))
    mask = torch.ones(1, 2, dtype=torch.bool)
    out = loss._rollout_is_weights(log_ratio, mask)
    assert out.tolist() == pytest.approx([0.5, 3.0])


def test_rollout_is_weights_token_truncate():
    """token_truncate: each token's ratio is clamped at the threshold -- 3.0 -> 2.0."""
    loss = _grpo_loss(rollout_importance_sampling_mode='token_truncate',
                      rollout_importance_sampling_threshold=2.0)
    log_ratio = torch.log(torch.tensor([0.5, 3.0]))
    mask = torch.ones(1, 2, dtype=torch.bool)
    out = loss._rollout_is_weights(log_ratio, mask)
    assert out.tolist() == pytest.approx([0.5, 2.0])


def test_rollout_is_weights_token_mask():
    """token_mask: tokens whose ratio exceeds the threshold are dropped entirely (weight 0)."""
    loss = _grpo_loss(rollout_importance_sampling_mode='token_mask',
                      rollout_importance_sampling_threshold=2.0)
    log_ratio = torch.log(torch.tensor([0.5, 3.0]))
    mask = torch.ones(1, 2, dtype=torch.bool)
    out = loss._rollout_is_weights(log_ratio, mask)
    assert out.tolist() == pytest.approx([0.5, 0.0])


def test_rollout_is_weights_sequence_truncate():
    """sequence_truncate: ONE weight per sequence -- exp(mean(log ratio)) = sqrt(0.5*3) = sqrt(1.5),
    clamped at the threshold, broadcast to every token."""
    loss = _grpo_loss(rollout_importance_sampling_mode='sequence_truncate',
                      rollout_importance_sampling_threshold=2.0)
    log_ratio = torch.log(torch.tensor([0.5, 3.0])).unsqueeze(0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    out = loss._rollout_is_weights(log_ratio, mask)
    expected = math.sqrt(1.5)  # below the 2.0 threshold -> unclamped
    assert out.squeeze(0).tolist() == pytest.approx([expected, expected])


def test_rollout_is_weights_sequence_mask():
    """sequence_mask: per-token ratios survive only when the sequence-level ratio is within the
    threshold -- here sqrt(1.5) <= 2.0 so both tokens keep their ratios."""
    loss = _grpo_loss(rollout_importance_sampling_mode='sequence_mask',
                      rollout_importance_sampling_threshold=2.0)
    log_ratio = torch.log(torch.tensor([[0.5, 3.0], [4.0, 4.0]]))
    mask = torch.ones(2, 2, dtype=torch.bool)
    out = loss._rollout_is_weights(log_ratio, mask)
    assert out[0].tolist() == pytest.approx([0.5, 3.0])  # seq ratio sqrt(1.5) <= 2 -> kept
    assert out[1].tolist() == pytest.approx([0.0, 0.0])  # seq ratio 4.0 > 2 -> dropped


class TestImportanceSamplingLevel:
    """The policy-vs-old ratio the PPO term consumes, at three levels."""

    LOGPS = torch.tensor([[1.0, 2.0]])
    OLD = torch.tensor([[0.0, 0.0]])
    MASK = torch.ones(1, 2, dtype=torch.bool)

    def test_token_level_is_per_token_log_ratio(self):
        """token (default): per-token log ratio, +-5 clamped for stability."""
        loss = _grpo_loss(importance_sampling_level='token')
        out = loss._compute_log_importance_weights(self.LOGPS, self.OLD, self.MASK)
        assert out.tolist() == [[1.0, 2.0]]

    def test_sequence_level_is_mask_averaged(self):
        """sequence (GSPO): the mask-averaged log ratio, one [B, 1] weight that broadcasts to
        every token downstream (mean of [1, 2] = 1.5)."""
        loss = _grpo_loss(importance_sampling_level='sequence')
        out = loss._compute_log_importance_weights(self.LOGPS, self.OLD, self.MASK)
        assert out.tolist() == [[1.5]]

    def test_sequence_token_keeps_token_gradient_with_sequence_weight(self):
        """sequence_token (GSPO-token): VALUE equals the sequence weight, but the gradient flows
        through the per-token logp (logp - sg[logp] + sg[seq])."""
        loss = _grpo_loss(importance_sampling_level='sequence_token')
        logps = self.LOGPS.clone().requires_grad_(True)
        out = loss._compute_log_importance_weights(logps, self.OLD, self.MASK)
        assert out.tolist() == [[1.5, 1.5]]  # value = the detached sequence weight
        out.sum().backward()
        # gradient is the per-token one (ones), NOT zero (pure sequence) -- the GSPO trick.
        assert logps.grad.tolist() == [[1.0, 1.0]]

    def test_unknown_level_rejected(self):
        loss = _grpo_loss()
        loss.importance_sampling_level = 'bogus'
        with pytest.raises(ValueError, match='importance_sampling_level'):
            loss._compute_log_importance_weights(self.LOGPS, self.OLD, self.MASK)

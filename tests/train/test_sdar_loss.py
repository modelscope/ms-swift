"""Unit tests for SDAR (Self-Distilled Agentic RL) confidence-gated teacher distillation loss.

Validates ``swift.rl_core.advantage.compute_sdar_loss`` against a reference reimplementation of
SDAR's ``compute_sdar_loss`` (SDAR/verl/trainer/ppo/sdar_utils.py).

Run standalone:
    python tests/train/test_sdar_loss.py
Or with pytest:
    pytest tests/train/test_sdar_loss.py
"""
import torch

from swift.rl_core.advantage import compute_sdar_loss


def _reference_sdar(student_lp, teacher_lp, mask, gate_beta, loss_agg_mode='token-mean'):
    """Reference: mirrors SDAR compute_sdar_loss exactly."""
    teacher = teacher_lp.detach()
    delta = teacher - student_lp.detach()
    gate = torch.sigmoid(gate_beta * delta).detach()
    kl = teacher - student_lp
    gated = gate * kl
    if loss_agg_mode == 'token-mean':
        return (gated * mask).sum() / mask.sum().clamp(min=1.0)
    if loss_agg_mode == 'seq-mean-token-sum':
        return (gated * mask).sum(dim=-1).mean()
    if loss_agg_mode == 'seq-mean-token-mean':
        return ((gated * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)).mean()
    raise ValueError(loss_agg_mode)


def _make_inputs(seed=0):
    torch.manual_seed(seed)
    B, T = 4, 6
    mask = torch.ones(B, T)
    mask[0, 4:] = 0  # a couple of padded positions
    mask[2, 5:] = 0
    teacher_lp = torch.randn(B, T)
    student_lp = torch.randn(B, T)
    return student_lp, teacher_lp, mask


def test_matches_reference_default_beta():
    student_lp, teacher_lp, mask = _make_inputs()
    loss, _ = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=5.0)
    ref = _reference_sdar(student_lp, teacher_lp, mask, gate_beta=5.0)
    assert torch.allclose(loss, ref, atol=1e-6), (loss - ref).abs().max()


def test_matches_reference_other_beta():
    student_lp, teacher_lp, mask = _make_inputs(seed=1)
    loss, _ = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=1.0)
    ref = _reference_sdar(student_lp, teacher_lp, mask, gate_beta=1.0)
    assert torch.allclose(loss, ref, atol=1e-6), (loss - ref).abs().max()


def test_seq_mean_agg_modes_match_reference():
    student_lp, teacher_lp, mask = _make_inputs(seed=5)
    for mode in ('token-mean', 'seq-mean-token-sum', 'seq-mean-token-mean'):
        loss, _ = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=3.0, loss_agg_mode=mode)
        ref = _reference_sdar(student_lp, teacher_lp, mask, gate_beta=3.0, loss_agg_mode=mode)
        assert torch.allclose(loss, ref, atol=1e-6), (mode, (loss - ref).abs().max())


def test_token_mean_aggregation_formula():
    """Loss equals the masked token-mean of gate * (teacher - student)."""
    student_lp, teacher_lp, mask = _make_inputs(seed=2)
    beta = 5.0
    delta = teacher_lp - student_lp
    gate = torch.sigmoid(beta * delta)
    expected = (gate * (teacher_lp - student_lp) * mask).sum() / mask.sum()
    loss, _ = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=beta)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gate_metric_is_sigmoid_beta_delta():
    """Reported sdar/gate_mean equals the masked mean of sigmoid(beta * (teacher - student))."""
    student_lp, teacher_lp, mask = _make_inputs(seed=3)
    beta = 5.0
    gate = torch.sigmoid(beta * (teacher_lp - student_lp))
    expected_gate_mean = (gate * mask).sum() / mask.sum()
    _, metrics = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=beta)
    assert abs(metrics['sdar/gate_mean'] - expected_gate_mean.item()) < 1e-6


def test_zero_gap_gives_zero_loss():
    """teacher == student -> kl_per_token == 0 -> loss == 0 (gate is 0.5 everywhere)."""
    student_lp, _, mask = _make_inputs(seed=6)
    loss, metrics = compute_sdar_loss(student_lp, student_lp.clone(), mask, gate_beta=5.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)
    assert abs(metrics['sdar/gate_mean'] - 0.5) < 1e-6


def test_grad_only_through_student():
    """Gate and teacher are stop-gradient: grad flows ONLY through the student log-probs, and
    equals the token-mean derivative -gate*mask/mask_sum (i.e. the gate carries no gradient)."""
    student_lp, teacher_lp, mask = _make_inputs(seed=4)
    beta = 5.0
    student_lp = student_lp.clone().requires_grad_(True)
    teacher_lp = teacher_lp.clone().requires_grad_(True)
    loss, _ = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=beta)
    loss.backward()

    assert teacher_lp.grad is None or torch.allclose(teacher_lp.grad, torch.zeros_like(teacher_lp.grad))
    assert student_lp.grad is not None and student_lp.grad.abs().sum() > 0

    gate = torch.sigmoid(beta * (teacher_lp.detach() - student_lp.detach()))
    expected_grad = -(gate * mask) / mask.sum().clamp(min=1.0)
    assert torch.allclose(student_lp.grad, expected_grad, atol=1e-6), \
        (student_lp.grad - expected_grad).abs().max()


def test_higher_beta_sharpens_gate():
    """All-positive gap: higher beta pushes the gate toward 1 (mean increases)."""
    B, T = 2, 3
    mask = torch.ones(B, T)
    student_lp = torch.zeros(B, T)
    teacher_lp = torch.full((B, T), 0.5)  # delta = +0.5 everywhere
    _, m_low = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=1.0)
    _, m_high = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=10.0)
    assert m_high['sdar/gate_mean'] > m_low['sdar/gate_mean']
    assert m_high['sdar/gate_active_ratio'] >= m_low['sdar/gate_active_ratio']


def test_metrics_keys_present():
    student_lp, teacher_lp, mask = _make_inputs(seed=7)
    _, metrics = compute_sdar_loss(student_lp, teacher_lp, mask, gate_beta=5.0)
    for key in ('sdar/gate_mean', 'sdar/gate_active_ratio', 'sdar/teacher_gap_mean', 'sdar/loss'):
        assert key in metrics, key


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('All SDAR loss tests passed.')

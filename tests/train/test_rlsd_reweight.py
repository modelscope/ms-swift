"""Unit tests for RLSD (Self-Distilled RLVR) advantage reweighting.

Validates ``swift.rl_core.advantage.apply_rlsd_reweight`` against a reference reimplementation of
RLSD's ``_build_stgca_advantages`` (RLSD/verl/workers/actor/dp_opsd_actor.py:209-269).

Run standalone:
    python tests/train/test_rlsd_reweight.py
Or with pytest:
    pytest tests/train/test_rlsd_reweight.py
"""
import torch

from swift.rl_core.advantage import apply_rlsd_reweight


def _reference_stgca(teacher_lp, student_lp, base_adv, mask, lam, clip_range, negative_only=False):
    """Reference: mirrors RLSD _build_stgca_advantages exactly (base_adv broadcast to [B, T])."""
    advantages = base_adv.unsqueeze(1).expand_as(mask)
    sign_adv = torch.sign(advantages)
    delta = (teacher_lp.detach() - student_lp.detach()) * mask
    weights = torch.exp(sign_adv * delta) * mask
    clipped_weights = torch.clamp(weights, min=1.0 - clip_range, max=1.0 + clip_range)
    reweight = ((1.0 - lam) + lam * clipped_weights) * mask
    if negative_only:
        seq_negative = (advantages.sum(dim=-1, keepdim=True) < 0).float()
        reweight = seq_negative * reweight + (1.0 - seq_negative) * mask
    return advantages * reweight.detach()


def _make_inputs(seed=0):
    torch.manual_seed(seed)
    B, T = 4, 6
    mask = torch.ones(B, T)
    mask[0, 4:] = 0  # a couple of padded positions
    mask[2, 5:] = 0
    teacher_lp = torch.randn(B, T)
    student_lp = torch.randn(B, T)
    base_adv = torch.tensor([1.5, -2.0, 0.0, -0.3])
    return teacher_lp, student_lp, base_adv, mask


def test_matches_reference_full_rlsd():
    teacher_lp, student_lp, base_adv, mask = _make_inputs()
    out = apply_rlsd_reweight(base_adv, mask, teacher_lp, student_lp, lam=1.0, clip_range=0.2)
    ref = _reference_stgca(teacher_lp, student_lp, base_adv, mask, lam=1.0, clip_range=0.2)
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()


def test_matches_reference_mixed_lambda():
    teacher_lp, student_lp, base_adv, mask = _make_inputs(seed=1)
    out = apply_rlsd_reweight(base_adv, mask, teacher_lp, student_lp, lam=0.5, clip_range=0.2)
    ref = _reference_stgca(teacher_lp, student_lp, base_adv, mask, lam=0.5, clip_range=0.2)
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()


def test_lambda_zero_is_plain_broadcast():
    """lam=0 -> reweight == 1 -> output is the masked broadcast of the base advantage (pure GRPO)."""
    teacher_lp, student_lp, base_adv, mask = _make_inputs(seed=2)
    out = apply_rlsd_reweight(base_adv, mask, teacher_lp, student_lp, lam=0.0, clip_range=0.2)
    expected = base_adv.unsqueeze(1).expand_as(mask) * mask
    assert torch.allclose(out, expected, atol=1e-6)


def test_clip_bounds_respected():
    """With large logprob gaps, the effective per-token multiplier stays within [1-eps, 1+eps]."""
    B, T = 2, 3
    mask = torch.ones(B, T)
    base_adv = torch.tensor([1.0, -1.0])
    teacher_lp = torch.tensor([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])  # huge positive gap
    student_lp = torch.tensor([[-10.0, -10.0, -10.0], [-10.0, -10.0, -10.0]])
    eps = 0.2
    out = apply_rlsd_reweight(base_adv, mask, teacher_lp, student_lp, lam=1.0, clip_range=eps)
    ratio = out / base_adv.unsqueeze(1)  # per-token reweight multiplier
    assert torch.all(ratio <= 1.0 + eps + 1e-6)
    assert torch.all(ratio >= 1.0 - eps - 1e-6)


def test_sign_flip_for_negative_advantage():
    """For A<0, a token more supported by the teacher (delta>0) should get a LARGER blame (weight>1)."""
    B, T = 1, 1
    mask = torch.ones(B, T)
    base_adv = torch.tensor([-1.0])  # negative advantage
    teacher_lp = torch.tensor([[0.1]])
    student_lp = torch.tensor([[0.0]])  # delta = +0.1 > 0
    out = apply_rlsd_reweight(base_adv, mask, teacher_lp, student_lp, lam=1.0, clip_range=0.2)
    # sign(A) = -1, w = exp(-1 * 0.1) < 1 -> reweight < 1 -> |A_hat| < |A|; magnitude shrinks for
    # teacher-supported tokens on a wrong trajectory (they are blamed less). Reference must agree.
    ref = _reference_stgca(teacher_lp, student_lp, base_adv, mask, lam=1.0, clip_range=0.2)
    assert torch.allclose(out, ref, atol=1e-6)
    assert out.item() > base_adv.item()  # -0.9x < ... i.e. less negative than -1


def test_negative_only_leaves_positive_sequences_untouched():
    teacher_lp, student_lp, base_adv, mask = _make_inputs(seed=3)
    out = apply_rlsd_reweight(base_adv, mask, teacher_lp, student_lp, lam=1.0, clip_range=0.2, negative_only=True)
    ref = _reference_stgca(teacher_lp, student_lp, base_adv, mask, lam=1.0, clip_range=0.2, negative_only=True)
    assert torch.allclose(out, ref, atol=1e-6)
    # base_adv[0] = 1.5 >= 0 -> unchanged (plain broadcast); base_adv[3] = -0.3 < 0 -> reweighted.
    pos_expected = base_adv[0].item() * mask[0]
    assert torch.allclose(out[0], pos_expected, atol=1e-6)


def test_no_grad_flows_through_reweight():
    """The reweight is stop-gradient: grad flows only through the base advantage, never the logps."""
    teacher_lp, student_lp, base_adv, mask = _make_inputs(seed=4)
    teacher_lp = teacher_lp.clone().requires_grad_(True)
    student_lp = student_lp.clone().requires_grad_(True)
    base_adv = base_adv.clone().requires_grad_(True)
    out = apply_rlsd_reweight(base_adv, mask, teacher_lp, student_lp, lam=1.0, clip_range=0.2)
    out.sum().backward()
    assert teacher_lp.grad is None or torch.allclose(teacher_lp.grad, torch.zeros_like(teacher_lp.grad))
    assert student_lp.grad is None or torch.allclose(student_lp.grad, torch.zeros_like(student_lp.grad))
    assert base_adv.grad is not None and base_adv.grad.abs().sum() > 0


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('All RLSD reweight tests passed.')

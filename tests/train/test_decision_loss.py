# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from swift.loss.decision import DistillRPSLoss, ScoringLoss


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_extreme_wrong_prediction_keeps_gradient(dtype):
    logits = torch.tensor([[30., -30.]], dtype=dtype, requires_grad=True)
    outputs = SimpleNamespace(logits=logits, option_mask=torch.ones_like(logits, dtype=torch.bool))
    loss = DistillRPSLoss(None, None)(outputs, torch.tensor([[0., 1.]], dtype=dtype))
    loss.backward()
    torch.testing.assert_close(logits.grad.float(), torch.tensor([[1., -1.]]))
    torch.testing.assert_close(loss, torch.tensor(60.))


def test_masked_soft_targets_match_cross_entropy():
    dtype = torch.float32
    logits = torch.tensor([[3., -1., 1000.], [2., 1., -2.]], dtype=dtype, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    target = torch.tensor([[.25, .75, .5], [.25, .25, .5]], dtype=dtype)
    loss = ScoringLoss(None, None)._masked_soft_ce(logits, mask, target)
    expected = torch.stack(
        [F.cross_entropy(reference[i, mask[i]].float(), target[i, mask[i]].float()) for i in range(2)])
    loss.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(logits.grad, reference.grad)
    torch.testing.assert_close(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))


def test_empty_option_row_has_finite_zero_loss_and_gradient():
    logits = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    mask = torch.tensor([[False, False], [True, True]])
    target = torch.tensor([[.5, .5], [0., 1.]])
    loss = ScoringLoss(None, None)._masked_soft_ce(logits, mask, target)
    loss.sum().backward()
    assert loss[0].item() == 0.
    assert torch.isfinite(loss).all()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[0], torch.zeros(2))

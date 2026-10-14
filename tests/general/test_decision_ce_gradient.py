# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for #10354.

`_masked_soft_ce` used to compute ``log(softmax(...).clamp_min(1e-9))``, which
truncates the per-class loss to ``-log(1e-9)`` and zeros the gradient for any
correct class whose probability has already fallen below ``1e-9``. Replacing
the ``log(softmax + clamp)`` with ``log_softmax`` keeps the loss finite and
its gradient non-zero no matter how small the model's mass on the target
becomes.
"""
import torch
import unittest
from torch.nn.functional import one_hot

from swift.loss.decision import ScoringLoss
from swift.model.decision_head import masked_log_softmax, masked_softmax


def _ce_old(logits: torch.Tensor, option_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The pre-#10354 implementation, kept here as the regression baseline."""
    probs = masked_softmax(logits, option_mask)
    log_probs = torch.log(probs.clamp_min(1e-9))
    return -(target * log_probs * option_mask).sum(dim=-1)


def _ce_new(logits: torch.Tensor, option_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = masked_log_softmax(logits, option_mask)
    return -(target * log_probs * option_mask).sum(dim=-1)


class TestDecisionCEGradient(unittest.TestCase):

    def test_correct_class_with_tiny_probability_still_gets_gradient(self):
        """A one-hot target on a class whose logit is so low that softmax
        collapses to < 1e-9 must still receive a corrective gradient."""
        torch.manual_seed(0)
        # Target is class 0; classes 1/2/3 sit ~50 nats above, so the
        # target-class probability falls well below the 1e-9 clamp.
        logits = torch.tensor([[-50.0, 0.0, 0.0, 0.0]], requires_grad=True)
        option_mask = torch.tensor([[True, True, True, True]])
        target = one_hot(torch.tensor([0]), num_classes=4).float()

        new_loss = _ce_new(logits, option_mask, target)
        new_loss.sum().backward()
        self.assertTrue(torch.isfinite(new_loss).all(), 'new loss must be finite')
        # Gradient must be non-zero on the target class so training can
        # recover from the confidently-wrong state.
        self.assertGreater(float(logits.grad[0, 0].abs()), 0.0)

        logits2 = torch.tensor([[-50.0, 0.0, 0.0, 0.0]], requires_grad=True)
        old_loss = _ce_old(logits2, option_mask, target)
        old_loss.sum().backward()
        # The pre-fix loss is capped at -log(1e-9) ~ 20.72 with zero
        # gradient on the target class — the bug this issue reports.
        self.assertEqual(float(logits2.grad[0, 0]), 0.0, f'old CE leaked a non-zero grad on class 0: {logits2.grad}')

    def test_correct_class_loss_is_finite_under_extreme_logits(self):
        torch.manual_seed(1)
        logits = torch.tensor([[-1e3, 1.0, 2.0]], requires_grad=True)
        option_mask = torch.tensor([[True, True, True]])
        target = one_hot(torch.tensor([0]), num_classes=3).float()

        new_loss = _ce_new(logits, option_mask, target)
        # Pre-fix this was clipped to ~20.72 because softmax(class 0) < 1e-9.
        self.assertGreater(float(new_loss.item()), 20.72)
        self.assertTrue(torch.isfinite(new_loss).all())

    def test_pad_slots_contribute_zero(self):
        """masked_log_softmax must follow masked_softmax's nan_to_num(0.0)
        contract so the existing `* option_mask` guard keeps them at zero."""
        logits = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        option_mask = torch.tensor([[True, True, False, False]])
        log_probs = masked_log_softmax(logits, option_mask)
        self.assertEqual(log_probs.shape, logits.shape)
        self.assertEqual(log_probs[0, 2:].tolist(), [0.0, 0.0])
        # The two real options should sum (in prob-space) to 1; the log-sum-exp
        # is therefore 0, so log_softmax over the two real options returns the
        # exact log-softmax of a 2-class vector.
        self.assertAllClose(
            log_probs[0, :2],
            torch.log_softmax(torch.tensor([1.0, 2.0]), dim=-1),
        )

    def test_one_hot_target_matches_softmax_cross_entropy(self):
        """For a one-hot target the soft CE equals `-log p(target)`. The new
        loss must match the old loss whenever the chosen class is not yet
        in the clamped regime (which is the case for the sampled logits)."""
        torch.manual_seed(2)
        logits = torch.randn(2, 5)
        option_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
        target = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]], dtype=torch.float)

        new_loss = _ce_new(logits, option_mask, target)
        old_loss = _ce_old(logits, option_mask, target)
        self.assertAllClose(new_loss, old_loss, atol=1e-6)

    def test_scoring_loss_subclass_uses_unclamped_path(self):
        """Smoke test that the live `ScoringLoss._masked_soft_ce` matches the
        new implementation. The class is constructed via the small surface
        that does not pull in the full swift trainer stack."""

        class _Dummy:

            def __init__(self):
                self._ce = ScoringLoss.__dict__['_masked_soft_ce']

        def call(logits, option_mask, target):
            # Bound the function to the dummy instance so subclasses that
            # expect `self` keep working.
            return ScoringLoss._masked_soft_ce(_Dummy(), logits, option_mask, target)

        torch.manual_seed(3)
        logits = torch.full((1, 3), -80.0, requires_grad=True)
        option_mask = torch.tensor([[True, True, True]])
        target = torch.tensor([[1.0, 0.0, 0.0]])

        loss = call(logits, option_mask, target)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue((logits.grad.abs() > 0).all())

    def assertAllClose(self, a, b, atol=1e-7):
        torch.testing.assert_close(a, b, atol=atol, rtol=0)


if __name__ == '__main__':
    unittest.main()

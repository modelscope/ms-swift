# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
import unittest
from types import SimpleNamespace

from swift.rlhf_trainers.dpo_trainer import DPOTrainer


def _trainer(beta, tau):
    trainer = object.__new__(DPOTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device('cpu'))
    trainer.reference_free = False
    trainer.f_divergence_type = 'reverse_kl'
    trainer.beta = beta
    trainer.args = SimpleNamespace(discopop_tau=tau)
    return trainer


def _logps(margins, beta, dtype):
    # All four inputs are valid (nonpositive) sequence log probabilities.
    chosen = (torch.tensor(margins, dtype=dtype) - 200) / beta
    rejected = torch.full_like(chosen, -200 / beta)
    reference = torch.full_like(chosen, -100)
    return [value.detach().clone().requires_grad_() for value in (chosen, rejected, reference, reference)]


def _reference_loss(logps, beta, tau):
    chosen, rejected, ref_chosen, ref_rejected = logps
    logits = beta * (chosen - rejected - ref_chosen + ref_rejected)
    modulation = torch.sigmoid(logits / tau)
    # Original objective; the large-margin oracle uses FP64, where exp(100) is finite.
    return -F.logsigmoid(logits) * (1 - modulation) + torch.exp(-logits) * modulation


class TestDPODiscoPOPLoss(unittest.TestCase):

    def test_large_margins_match_fp64_loss_and_gradients(self):
        for beta in (0.1, 0.5):
            for tau in (0.05, 0.5, 1.0, 2.0):
                with self.subTest(beta=beta, tau=tau):
                    logps = _logps([-100, -50, 0, 50, 100], beta, torch.float32)
                    reference_logps = [value.detach().double().requires_grad_() for value in logps]
                    losses, chosen_rewards, rejected_rewards = DPOTrainer.dpo_loss(
                        _trainer(beta, tau), *logps, loss_type='discopop')
                    expected = _reference_loss(reference_logps, beta, tau)
                    gradients = torch.autograd.grad(losses.sum(), logps)
                    expected_gradients = torch.autograd.grad(expected.sum(), reference_logps)

                    self.assertTrue(torch.isfinite(losses).all())
                    torch.testing.assert_close(losses.double(), expected, rtol=1e-5, atol=1e-6)
                    for actual, reference in zip(gradients, expected_gradients):
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(chosen_rewards, beta * (logps[0] - logps[2]).detach())
                    torch.testing.assert_close(rejected_rewards, beta * (logps[1] - logps[3]).detach())

    def test_ordinary_margins_preserve_objective(self):
        for dtype in (torch.float32, torch.float64):
            for tau in (0.05, 0.5, 1.0, 2.0):
                with self.subTest(dtype=dtype, tau=tau):
                    beta = 0.1
                    logps = _logps([-5, -1, 0, 1, 5], beta, dtype)
                    reference_logps = [value.detach().clone().requires_grad_() for value in logps]
                    losses, _, _ = DPOTrainer.dpo_loss(_trainer(beta, tau), *logps, loss_type='discopop')
                    expected = _reference_loss(reference_logps, beta, tau)

                    torch.testing.assert_close(losses, expected)
                    gradients = torch.autograd.grad(losses.sum(), logps)
                    expected_gradients = torch.autograd.grad(expected.sum(), reference_logps)
                    for actual, reference in zip(gradients, expected_gradients):
                        torch.testing.assert_close(actual, reference)

    def test_bfloat16_large_negative_margin_has_finite_gradient(self):
        chosen = torch.tensor([-1000.], dtype=torch.bfloat16, requires_grad=True)
        rejected = torch.zeros_like(chosen)
        reference = torch.full_like(chosen, -100)
        losses, _, _ = DPOTrainer.dpo_loss(
            _trainer(0.1, 0.05), chosen, rejected, reference, reference, loss_type='discopop')
        gradient, = torch.autograd.grad(losses.sum(), chosen)

        torch.testing.assert_close(losses, torch.tensor([100.], dtype=torch.bfloat16))
        torch.testing.assert_close(gradient, torch.tensor([-0.1], dtype=torch.bfloat16))


if __name__ == '__main__':
    unittest.main()

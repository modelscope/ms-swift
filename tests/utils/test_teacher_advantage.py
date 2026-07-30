"""Tests for masking teacher log-probability signals outside completion tokens."""

import torch
import unittest

from swift.rl_core.advantage import compute_teacher_kl_per_token, compute_teacher_logratio


class TestTeacherAdvantageMasking(unittest.TestCase):

    def test_teacher_kl_masks_padding_before_exp(self):
        teacher_logps = torch.tensor([[-2.0, 0.0, -3.0, 0.0]])
        policy_logps = torch.tensor([[-3.0, -1e10, -1.0, -1e10]])
        completion_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)

        result = compute_teacher_kl_per_token(teacher_logps, policy_logps, completion_mask)
        active_delta = torch.tensor([1.0, -2.0])
        expected_active = torch.exp(active_delta) - active_delta - 1

        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[completion_mask], expected_active)
        torch.testing.assert_close(result[~completion_mask], torch.zeros(2))

    def test_teacher_helpers_ignore_nonfinite_padding(self):
        teacher_logps = torch.tensor([[-2.0, float('nan'), 0.0]])
        policy_logps = torch.tensor([[-3.0, float('inf'), -1e10]])
        completion_mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)

        teacher_kl = compute_teacher_kl_per_token(teacher_logps, policy_logps, completion_mask)
        logratio = compute_teacher_logratio(teacher_logps, policy_logps, completion_mask)

        self.assertTrue(torch.isfinite(teacher_kl).all())
        self.assertTrue(torch.isfinite(logratio).all())
        torch.testing.assert_close(teacher_kl, torch.tensor([[torch.e - 2.0, 0.0, 0.0]]))
        torch.testing.assert_close(logratio, torch.tensor([[1.0, 0.0, 0.0]]))


if __name__ == '__main__':
    unittest.main()

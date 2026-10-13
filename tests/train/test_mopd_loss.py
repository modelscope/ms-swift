# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import torch.nn.functional as F
import unittest

from swift.rlhf_trainers.gkd_loss import TeacherOutput, mopd_gkd_loss, topk_mopd_loss


class TestMOPDLoss(unittest.TestCase):

    def test_formula_mask_and_teacher_detachment(self):
        student = torch.tensor([[[.2, .1], [.4, .3]]], dtype=torch.float64).log().requires_grad_()
        teacher = torch.tensor([[[.3, .2], [.2, .1]]], dtype=torch.float64).log().requires_grad_()
        mask = torch.tensor([[True, False]])
        loss = topk_mopd_loss(student, teacher, mask)
        expected = sum(s * math.log(s / t) - s + t for s, t in ((.2, .3), (.1, .2)))
        torch.testing.assert_close(loss, torch.tensor([[expected, 0.]], dtype=torch.float64))
        loss.sum().backward()
        expected_grad = torch.tensor([[[.2 * math.log(.2 / .3), .1 * math.log(.1 / .2)], [0., 0.]]],
                                     dtype=torch.float64)
        torch.testing.assert_close(student.grad, expected_grad)
        self.assertIsNone(teacher.grad)

    def test_full_support_matches_reverse_kl(self):
        torch.manual_seed(42)
        student = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn_like(student).log_softmax(-1)
        actual = topk_mopd_loss(student.log_softmax(-1), teacher)
        expected = F.kl_div(teacher, student.log_softmax(-1), log_target=True, reduction='none').sum(-1)
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual.sum(), student, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected.sum(), student)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_local_and_api_teacher_agree_without_topk_renormalization(self):
        torch.manual_seed(7)
        student = torch.randn(2, 4, 13, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn_like(student)
        labels = torch.tensor([[-100, 1, 2, -100], [1, -100, 2, 3]])
        local = TeacherOutput(full_logits=teacher).to_topk(3, normalize=True)
        values, indices = teacher.log_softmax(-1).topk(3, dim=-1)
        api = TeacherOutput(topk_logprobs=values, topk_indices=indices)
        actual, count = mopd_gkd_loss(student, local, labels)
        expected, expected_count = mopd_gkd_loss(student, api, labels)
        self.assertEqual(count.item(), 5)
        self.assertEqual(count.item(), expected_count.item())
        torch.testing.assert_close(actual, expected)
        self.assertTrue((local.topk_logprobs.exp().sum(-1) < 1.).all())
        s_prob = student.softmax(-1).gather(-1, indices)[labels != -100]
        t_prob = teacher.softmax(-1).gather(-1, indices)[labels != -100]
        reference = (s_prob * (s_prob / t_prob).log() - s_prob + t_prob).sum()
        torch.testing.assert_close(actual, reference)
        actual_grad, = torch.autograd.grad(actual, student, retain_graph=True)
        expected_grad, = torch.autograd.grad(reference, student)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_gradcheck_and_gradients_outside_topk(self):
        student = torch.tensor([[[.2, .3, .5]]], dtype=torch.float64).log().requires_grad_()
        teacher = TeacherOutput(
            topk_logprobs=torch.tensor([[[.4]]], dtype=torch.float64).log(), topk_indices=torch.tensor([[[0]]]))
        labels = torch.ones(1, 1, dtype=torch.long)
        self.assertTrue(torch.autograd.gradcheck(lambda x: mopd_gkd_loss(x, teacher, labels)[0], (student, )))
        loss, _ = mopd_gkd_loss(student, teacher, labels)
        loss.backward()
        self.assertGreater(student.grad[0, 0, 1].abs().item(), 0.)
        self.assertGreater(student.grad[0, 0, 2].abs().item(), 0.)

    def test_chunking_and_independent_teacher_mask(self):
        torch.manual_seed(11)
        student = torch.randn(1, 7, 9, dtype=torch.float64, requires_grad=True)
        teacher = TeacherOutput(
            full_logits=torch.randn(1, 8, 9, dtype=torch.float64),
            labels=torch.tensor([[-100, -100, 1, 2, 3, 4, -100, -100]])).to_topk(
                4, normalize=True)
        labels = torch.tensor([[-100, 1, 2, 3, 4, -100, -100]])
        reference, _ = mopd_gkd_loss(student, teacher, labels, chunk_size=512)
        reference_grad, = torch.autograd.grad(reference, student, retain_graph=True)
        for chunk_size in (1, 3):
            actual, count = mopd_gkd_loss(student, teacher, labels, chunk_size=chunk_size)
            self.assertEqual(count.item(), 4)
            torch.testing.assert_close(actual, reference)
            gradient, = torch.autograd.grad(actual, student, retain_graph=True)
            torch.testing.assert_close(gradient, reference_grad)

    def test_low_precision_matches_fp64_reference(self):
        torch.manual_seed(13)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                student = torch.randn(2, 4, 17, dtype=dtype, requires_grad=True)
                teacher_logits = torch.randn_like(student)
                labels = torch.tensor([[-100, 1, 2, -100], [1, 2, 3, 4]])
                teacher = TeacherOutput(full_logits=teacher_logits).to_topk(5, normalize=True)
                actual, _ = mopd_gkd_loss(student, teacher, labels)
                reference_student = student.detach().double().requires_grad_()
                reference_teacher = TeacherOutput(full_logits=teacher_logits.double()).to_topk(5, normalize=True)
                reference, _ = mopd_gkd_loss(reference_student, reference_teacher, labels)
                torch.testing.assert_close(actual.double(), reference, atol=2e-6, rtol=2e-6)
                gradient, = torch.autograd.grad(actual, student)
                reference_gradient, = torch.autograd.grad(reference, reference_student)
                torch.testing.assert_close(gradient, reference_gradient.to(dtype), atol=2e-6, rtol=2e-3)

    def test_missing_api_entries_and_empty_batches(self):
        student = torch.tensor([[[1., 2., 3.], [2., 1., 3.]]], requires_grad=True)
        teacher = TeacherOutput(
            topk_logprobs=torch.tensor([[[-1., float('-inf')], [float('-inf'), float('-inf')]]]),
            topk_indices=torch.tensor([[[2, 0], [0, 0]]]))
        labels = torch.ones(1, 2, dtype=torch.long)
        actual, count = mopd_gkd_loss(student, teacher, labels)
        self.assertEqual(count.item(), 1)
        reference = topk_mopd_loss(student.log_softmax(-1)[:, :1, 2:], teacher.topk_logprobs[:, :1, :1]).sum()
        torch.testing.assert_close(actual, reference)
        actual.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        torch.testing.assert_close(student.grad[:, 1], torch.zeros_like(student.grad[:, 1]))
        for empty_labels in (torch.full_like(labels, -100), ):
            student.grad = None
            empty, count = mopd_gkd_loss(student, teacher, empty_labels)
            self.assertEqual(count.item(), 0)
            self.assertEqual(empty.item(), 0.)
            empty.backward()
            torch.testing.assert_close(student.grad, torch.zeros_like(student.grad))

    def test_equal_and_near_equal_distributions(self):
        torch.manual_seed(19)
        for dtype in (torch.float32, torch.float64):
            logits = torch.randn(2, 3, 29, dtype=dtype)
            t_log = logits.log_softmax(-1)
            s_log = t_log.detach().clone().requires_grad_()
            loss = topk_mopd_loss(s_log, t_log)
            torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0., rtol=0.)
            loss.sum().backward()
            torch.testing.assert_close(s_log.grad, torch.zeros_like(s_log), atol=0., rtol=0.)
            near = topk_mopd_loss((logits + torch.randn_like(logits) * 1e-5).log_softmax(-1), t_log)
            self.assertTrue(torch.isfinite(near).all())
            self.assertGreaterEqual(near.min().item(), -2e-7)

    def test_invalid_shapes_and_missing_topk(self):
        with self.assertRaises(ValueError):
            topk_mopd_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))
        with self.assertRaises(ValueError):
            topk_mopd_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 3))
        with self.assertRaises(ValueError):
            mopd_gkd_loss(torch.zeros(1, 2, 3), TeacherOutput(full_logits=torch.zeros(1, 2, 3)), torch.ones(1, 2))


if __name__ == '__main__':
    unittest.main()

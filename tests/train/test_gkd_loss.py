# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
import unittest

from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss


class TestGKDLoss(unittest.TestCase):

    def test_empty_loss_dtype_and_vocab_alignment(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for teacher_vocab in (5, 8, 11):
                with self.subTest(dtype=dtype, teacher_vocab=teacher_vocab):
                    student = torch.randn(1, 4, 8, dtype=dtype, requires_grad=True)
                    teacher = TeacherOutput(full_logits=torch.randn(1, 4, teacher_vocab, dtype=dtype))
                    labels = torch.full((1, 4), -100)
                    total, count = gkd_loss(student, teacher, labels, 0.5, 1.)
                    self.assertEqual(total.dtype, dtype)
                    self.assertEqual(total.device, student.device)
                    self.assertEqual(total.item(), 0.)
                    self.assertEqual(count.item(), 0)
                    total.backward()
                    torch.testing.assert_close(student.grad, torch.zeros_like(student))
                    with torch.no_grad():
                        evaluation_loss, _ = gkd_loss(student, teacher, labels, 0.5, 1.)
                    self.assertFalse(evaluation_loss.requires_grad)
                    self.assertEqual(evaluation_loss.item(), 0.)

    def test_empty_microbatch_preserves_accumulated_update(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 8)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
        inputs = torch.randn(1, 4, 3)
        teacher = TeacherOutput(full_logits=torch.randn(1, 4, 8))
        valid_labels = torch.tensor([[-100, -100, 1, 2]])
        for _ in range(2):
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            for labels in (valid_labels, torch.full_like(valid_labels, -100)):
                total, _ = gkd_loss(model(inputs), teacher, labels, 0.5, 1.)
                (total / 2).backward()
            expected, _ = gkd_loss(reference(inputs), teacher, valid_labels, 0.5, 1.)
            (expected / 2).backward()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad)
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected)

    def test_empty_active_tokens_backward(self):
        for beta in (0., 0.5, 1.):
            for mode in ('full', 'topk', 'uncovered_topk'):
                with self.subTest(beta=beta, mode=mode):
                    student = torch.randn(1, 4, 8, requires_grad=True)
                    labels = torch.full((1, 4), -100)
                    teacher_logits = torch.randn_like(student)
                    if mode == 'full':
                        teacher = TeacherOutput(full_logits=teacher_logits)
                    else:
                        values, indices = teacher_logits.topk(3, dim=-1)
                        if mode == 'uncovered_topk':
                            labels.fill_(1)
                            values.fill_(float('-inf'))
                        teacher = TeacherOutput(topk_logprobs=values, topk_indices=indices)
                    total, count = gkd_loss(student, teacher, labels, beta, temperature=2.)
                    self.assertEqual(count.item(), 0)
                    self.assertEqual(total.item(), 0.)
                    total.backward()
                    self.assertIsNotNone(student.grad)
                    torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_empty_partition_preserves_model_gradients(self):
        model = torch.nn.Linear(3, 8)
        logits = model(torch.ones(1, 4, 3))
        total, count = gkd_loss(logits, TeacherOutput(full_logits=torch.zeros_like(logits)), torch.full((1, 4), -100),
                                0.5, 1.)
        (total / count.clamp_min(1)).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_valid_tokens_match_reference(self):
        for beta in (0., 0.5, 1.):
            with self.subTest(beta=beta):
                torch.manual_seed(42)
                student = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
                teacher = torch.randn_like(student)
                labels = torch.tensor([[-100, 1, -100, 2]])
                total, count = gkd_loss(student, TeacherOutput(full_logits=teacher), labels, beta, 2., chunk_size=1)
                s_log = (student[labels != -100] / 2.).log_softmax(-1)
                t_log = (teacher[labels != -100] / 2.).log_softmax(-1)
                if beta == 0.:
                    expected = (t_log.exp() * (t_log - s_log)).sum()
                elif beta == 1.:
                    expected = (s_log.exp() * (s_log - t_log)).sum()
                else:
                    mixture_log = ((1 - beta) * s_log.exp() + beta * t_log.exp()).log()
                    expected = (beta * t_log.exp() * (t_log - mixture_log) + (1 - beta) * s_log.exp() *
                                (s_log - mixture_log)).sum()
                self.assertEqual(count.item(), 2)
                torch.testing.assert_close(total, expected)
                actual_grad, = torch.autograd.grad(total, student, retain_graph=True)
                expected_grad, = torch.autograd.grad(expected, student)
                torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == '__main__':
    unittest.main()

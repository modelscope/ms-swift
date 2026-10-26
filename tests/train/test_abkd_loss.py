# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss


def reference_abkd(student_logits, teacher_logits, alpha, beta, temperature=1.):
    """Evaluate the paper's probability-space definition independently in FP64."""
    log_q = (student_logits.double() / temperature).log_softmax(-1)
    log_p = (teacher_logits.double() / temperature).log_softmax(-1)
    p, q = log_p.exp(), log_q.exp()
    if alpha == 0. and beta == 0.:
        values = 0.5 * (log_p - log_q).square()
    elif alpha == 0.:
        values = (q.pow(beta) * (beta * (log_q - log_p) - 1) + p.pow(beta)) / beta**2
    elif beta == 0.:
        p_alpha = p.pow(alpha)
        values = (alpha * torch.special.xlogy(p_alpha, p) - p_alpha * (alpha * log_q + 1) + q.pow(alpha)) / alpha**2
    elif alpha + beta == 0.:
        values = (alpha * (log_q - log_p) + (p / q).pow(alpha) - 1) / alpha**2
    else:
        total = alpha + beta
        values = (alpha / total * p.pow(total) + beta / total * q.pow(total) - p.pow(alpha) * q.pow(beta)) / (
            alpha * beta)
    return values.sum()


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('topk', [None, 7])
@pytest.mark.parametrize('temperature', [0.8, 2.])
@pytest.mark.parametrize('alpha,beta', [(0.2, 0.7), (1.2, -0.2), (-0.3, 0.8), (0., 0.7), (0.8, 0.), (0.4, -0.4),
                                        (0., 0.)])
def test_abkd_matches_paper_loss_and_student_gradients(device, dtype, topk, temperature, alpha, beta):
    torch.manual_seed(42)
    student = torch.randn(2, 7, 23, device=device, dtype=dtype, requires_grad=True)
    teacher_logits = torch.randn_like(student)
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4, -100], [-100, 1, 2, 3, -100, -100, -100]], device=device)
    teacher = TeacherOutput(full_logits=teacher_logits, labels=labels)
    if topk is not None:
        teacher = teacher.to_topk(topk)

    total, count = gkd_loss(
        student, teacher, labels, 0.5, temperature, chunk_size=3, loss_type='abkd', abkd_alpha=alpha, abkd_beta=beta)

    reference_student = student.detach().double().requires_grad_()
    active_student = reference_student[labels != -100]
    if topk is None:
        active_teacher = teacher_logits[labels != -100]
    else:
        active_student = active_student.gather(-1, teacher.topk_indices[labels != -100])
        active_teacher = teacher.topk_logprobs[labels != -100]
    expected = reference_abkd(active_student, active_teacher, alpha, beta, temperature)
    actual_gradient, = torch.autograd.grad(total, student)
    expected_gradient, = torch.autograd.grad(expected, reference_student)

    assert count.item() == 7
    assert total.dtype == torch.float32
    torch.testing.assert_close(total.double(), expected, rtol=3e-5, atol=3e-6)
    gradient_rtol = 8e-3 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(actual_gradient.double(), expected_gradient, rtol=gradient_rtol, atol=2e-6)
    torch.testing.assert_close(actual_gradient[labels == -100], torch.zeros_like(actual_gradient[labels == -100]))


@pytest.mark.parametrize('topk', [None, 5])
def test_abkd_aligns_different_teacher_and_student_prompts(device, topk):
    torch.manual_seed(7)
    student = torch.randn(1, 8, 19, device=device, requires_grad=True)
    teacher_logits = torch.randn(1, 10, 19, device=device)
    labels = torch.tensor([[-100, -100, -100, 1, 2, 3, 4, -100]], device=device)
    teacher_labels = torch.tensor([[-100, -100, -100, -100, -100, 1, 2, 3, 4, -100]], device=device)
    teacher = TeacherOutput(full_logits=teacher_logits, labels=teacher_labels)
    if topk is not None:
        teacher = teacher.to_topk(topk)
    total, count = gkd_loss(student, teacher, labels, 0.5, 1.5, loss_type='abkd')
    active_student = student[labels != -100]
    if topk is None:
        active_teacher = teacher_logits[teacher_labels != -100]
    else:
        active_student = active_student.gather(-1, teacher.topk_indices[teacher_labels != -100])
        active_teacher = teacher.topk_logprobs[teacher_labels != -100]
    expected = reference_abkd(active_student, active_teacher, 0.2, 0.7, 1.5)
    assert count.item() == 4
    torch.testing.assert_close(total.double(), expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize('chunk_size', [1, 3, 512])
def test_abkd_chunking_preserves_loss_and_gradient(device, chunk_size):
    torch.manual_seed(11)
    student = torch.randn(2, 9, 31, device=device, requires_grad=True)
    teacher = TeacherOutput(full_logits=torch.randn_like(student))
    labels = torch.ones(2, 9, device=device, dtype=torch.long)
    total, _ = gkd_loss(student, teacher, labels, 0.5, 0.7, chunk_size=chunk_size, loss_type='abkd')
    expected = reference_abkd(student, teacher.full_logits, 0.2, 0.7, 0.7)
    actual_gradient, = torch.autograd.grad(total, student, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, student)
    torch.testing.assert_close(total.double(), expected, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_abkd_peaked_distributions_remain_accurate(device, dtype):
    torch.manual_seed(29)
    student = (torch.randn(1, 4, 1024, device=device) * 8).to(dtype).requires_grad_()
    teacher = torch.randn_like(student) * 8
    labels = torch.ones(1, 4, device=device, dtype=torch.long)
    total, _ = gkd_loss(student, TeacherOutput(full_logits=teacher), labels, 0.5, 1., loss_type='abkd')
    reference_student = student.detach().double().requires_grad_()
    expected = reference_abkd(reference_student, teacher, 0.2, 0.7)
    actual_gradient, = torch.autograd.grad(total, student)
    expected_gradient, = torch.autograd.grad(expected, reference_student)
    torch.testing.assert_close(total.double(), expected, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(actual_gradient.double(), expected_gradient, rtol=8e-3, atol=2e-6)


@pytest.mark.parametrize('topk', [None, 3])
def test_abkd_empty_microbatch_preserves_accumulated_gradient(device, topk):
    torch.manual_seed(5)
    student = torch.randn(1, 4, 11, device=device, requires_grad=True)
    teacher = TeacherOutput(full_logits=torch.randn_like(student))
    if topk is not None:
        teacher = teacher.to_topk(topk)
    labels = torch.tensor([[-100, 1, 2, -100]], device=device)
    total, _ = gkd_loss(student, teacher, labels, 0.5, 1., loss_type='abkd')
    total.backward()
    before = student.grad.clone()
    empty_total, count = gkd_loss(student, teacher, torch.full_like(labels, -100), 0.5, 1., loss_type='abkd')
    empty_total.backward()
    assert count.item() == 0
    assert empty_total.item() == 0.
    torch.testing.assert_close(student.grad, before)


@pytest.mark.parametrize('alpha,beta', [(0.2, 0.7), (0.2, 0.), (1., 0.)])
def test_abkd_partial_teacher_api_coverage(device, alpha, beta):
    torch.manual_seed(31)
    student = torch.randn(1, 3, 11, device=device, requires_grad=True)
    values = torch.tensor([[[-0.2, -1., float('-inf')], [-0.4, float('-inf'), -1.3],
                            [float('-inf'), float('-inf'), float('-inf')]]],
                          device=device)
    indices = torch.tensor([[[1, 4, 7], [2, 3, 9], [0, 1, 2]]], device=device)
    labels = torch.ones(1, 3, device=device, dtype=torch.long)
    total, count = gkd_loss(
        student,
        TeacherOutput(topk_logprobs=values, topk_indices=indices),
        labels,
        0.5,
        0.8,
        loss_type='abkd',
        abkd_alpha=alpha,
        abkd_beta=beta)
    reference_student = student.detach().double().requires_grad_()
    selected_student = reference_student[:, :2].gather(-1, indices[:, :2])
    expected = reference_abkd(selected_student, values[:, :2], alpha, beta, 0.8)
    actual_gradient, = torch.autograd.grad(total, student)
    expected_gradient, = torch.autograd.grad(expected, reference_student)
    assert count.item() == 2
    torch.testing.assert_close(total.double(), expected, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(actual_gradient.double(), expected_gradient, rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize('teacher_vocab,alpha,beta', [(11, 0.2, 0.7), (11, 0., 0.7), (11, 0.8, 0.), (11, 0.4, -0.4),
                                                      (11, 0., 0.), (29, 0.2, 0.7)])
def test_abkd_vocab_alignment_preserves_both_gradient_paths(device, teacher_vocab, alpha, beta):
    torch.manual_seed(19)
    student = torch.randn(1, 4, 19, device=device, requires_grad=True)
    teacher = torch.randn(1, 4, teacher_vocab, device=device)
    labels = torch.tensor([[-100, 1, 2, 3]], device=device)
    actual, count = gkd_loss(
        student,
        TeacherOutput(full_logits=teacher),
        labels,
        0.5,
        0.8,
        loss_type='abkd',
        abkd_alpha=alpha,
        abkd_beta=beta)

    reference_student = student.detach().double().requires_grad_()
    active_student = reference_student[labels != -100]
    active_teacher = teacher[labels != -100].double()
    if teacher_vocab < 19:
        # Vocabulary alignment copies the student's extra logits into the teacher.
        # Those logits contribute through both distributions, including normalization.
        active_teacher = torch.cat([active_teacher, active_student[:, teacher_vocab:]], dim=-1)
    else:
        active_student = torch.cat([active_student, active_teacher[:, 19:]], dim=-1)
    expected = reference_abkd(active_student, active_teacher, alpha, beta, 0.8)
    actual_gradient, = torch.autograd.grad(actual, student)
    expected_gradient, = torch.autograd.grad(expected, reference_student)
    assert count.item() == 3
    torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual_gradient.double(), expected_gradient, rtol=1e-4, atol=2e-6)

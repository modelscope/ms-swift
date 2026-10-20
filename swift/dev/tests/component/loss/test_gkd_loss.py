# Copyright (c) ModelScope Contributors. All rights reserved.
"""``swift.rlhf_trainers.gkd_loss``: the JSD math and the pipeline around it.

``jsd_loss`` is the single place the KL/JSD math lives, with ``beta`` interpolating forward KL (0),
reverse KL (1) and their geometric mixture in between; the three readings are pinned against the same
primitives the implementation uses. ``gkd_loss`` is the pipeline -- mask by labels, align student and
teacher vocabularies, scale by temperature -- and returns ``(total, num_valid)`` *unnormalized*,
because the caller owns the reduction (HF divides, Megatron all-reduces first).

The top-k teacher path has its own hazard: a position whose top-k logprobs are all ``-inf`` carries no
signal and is dropped from both sides, so ``num_valid`` can be smaller than the label count.
"""
import pytest
import torch
import torch.nn.functional as F

from swift.rlhf_trainers.gkd_loss import (
    TeacherOutput,
    _align_vocab,
    default_kl_div,
    extract_active,
    gkd_loss,
    jsd_loss,
)


def _log_softmax(x):
    return F.log_softmax(x, dim=-1)


# ---------------------------------------------------------------------------
# jsd_loss
# ---------------------------------------------------------------------------

def test_forward_kl_at_beta_zero():
    student = torch.tensor([[1., 0.]])
    teacher = torch.tensor([[0., 1.]])
    expected = float(default_kl_div(_log_softmax(student), _log_softmax(teacher)))
    assert float(jsd_loss(student, teacher, 0.0)) == pytest.approx(expected)
    assert expected == pytest.approx(0.46211716532707214, rel=1e-5)


def test_reverse_kl_at_beta_one():
    student = torch.tensor([[1., 0.]])
    teacher = torch.tensor([[0., 1.]])
    expected = float(default_kl_div(_log_softmax(teacher), _log_softmax(student)))
    assert float(jsd_loss(student, teacher, 1.0)) == pytest.approx(expected)
    assert expected == pytest.approx(0.46211716532707214, rel=1e-5)


def test_a_geometric_mixture_in_between():
    student = torch.tensor([[1., 0.]])
    teacher = torch.tensor([[0., 1.]])
    assert float(jsd_loss(student, teacher, 0.5)) == pytest.approx(0.1109439879655838, rel=1e-5)


def test_identical_distributions_score_zero():
    logits = torch.tensor([[0.3, -1.0, 2.0], [1.0, 0.0, -0.5]])
    assert float(jsd_loss(logits, logits.clone(), 0.5)) == pytest.approx(0.0, abs=1e-6)


def test_an_empty_batch_returns_a_zero_scalar():
    """An empty CP partition must still produce a scalar, so the all-reduce cannot hang."""
    total = jsd_loss(torch.zeros(0, 3), torch.zeros(0, 3), 0.5)
    assert total.dim() == 0
    assert float(total) == 0.0


def test_chunking_does_not_change_the_result():
    student = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    teacher = torch.randn(4, 3, generator=torch.Generator().manual_seed(1))
    assert float(jsd_loss(student, teacher, 0.5, chunk_size=2)) == pytest.approx(
        float(jsd_loss(student, teacher, 0.5, chunk_size=512)), rel=1e-6)


def test_default_kl_div_is_kl_of_target_given_input():
    input_log = _log_softmax(torch.tensor([[0.2, -0.5, 1.0]]))
    target_log = _log_softmax(torch.tensor([[-1.0, 0.3, 2.0]]))
    expected = (target_log.exp() * (target_log - input_log)).sum()
    assert float(default_kl_div(input_log, target_log)) == pytest.approx(float(expected))


# ---------------------------------------------------------------------------
# gkd_loss: the pipeline
# ---------------------------------------------------------------------------

STUDENT = torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(2))
TEACHER = torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(3))
LABELS = torch.tensor([[1, -100, 2]])
ACTIVE = LABELS != -100


def _topk_teacher(k=2):
    values, indices = torch.topk(TEACHER, k=k, dim=-1)
    return TeacherOutput(topk_logprobs=values, topk_indices=indices, labels=LABELS)


def test_the_full_vocab_path_masks_padding_and_reports_the_valid_count():
    total, num_valid = gkd_loss(
        STUDENT, TeacherOutput(full_logits=TEACHER, labels=LABELS), LABELS, beta=0.0, temperature=1.0)
    assert int(num_valid) == 2
    expected = float(jsd_loss(STUDENT[ACTIVE], TEACHER[ACTIVE], 0.0))
    assert float(total) == pytest.approx(expected, rel=1e-6)


def test_temperature_scales_the_logits_before_the_jsd():
    total_1, _ = gkd_loss(STUDENT, TeacherOutput(full_logits=TEACHER, labels=LABELS), LABELS, 0.0, 1.0)
    total_2, _ = gkd_loss(STUDENT, TeacherOutput(full_logits=TEACHER, labels=LABELS), LABELS, 0.0, 2.0)
    expected = float(jsd_loss(STUDENT[ACTIVE] / 2, TEACHER[ACTIVE] / 2, 0.0))
    assert float(total_2) == pytest.approx(expected, rel=1e-6)
    assert float(total_1) != pytest.approx(float(total_2))


def test_the_topk_path_gathers_the_students_matching_logits():
    teacher = _topk_teacher()
    total, num_valid = gkd_loss(STUDENT, teacher, LABELS, beta=0.0, temperature=1.0)
    assert int(num_valid) == 2
    gathered = torch.gather(STUDENT[ACTIVE], dim=-1, index=teacher.topk_indices[ACTIVE])
    expected = float(jsd_loss(gathered, teacher.topk_logprobs[ACTIVE], 0.0))
    assert float(total) == pytest.approx(expected, rel=1e-6)


def test_a_topk_position_with_no_signal_is_dropped():
    """All ``-inf`` means the teacher covered none of the student's support there, so the position is
    removed from both sides rather than contributing a fabricated loss."""
    teacher = _topk_teacher()
    teacher.topk_logprobs[0, 0] = float('-inf')  # position 0 is an active one
    _, num_valid = gkd_loss(STUDENT, teacher, LABELS, beta=0.0, temperature=1.0)
    assert int(num_valid) == 1


def test_converting_full_logits_to_topk_matches_an_explicit_topk_teacher():
    converted = gkd_loss(
        STUDENT, TeacherOutput(full_logits=TEACHER, labels=LABELS).to_topk(2), LABELS, 0.0, 1.0)
    explicit = gkd_loss(STUDENT, _topk_teacher(2), LABELS, 0.0, 1.0)
    assert int(converted[1]) == int(explicit[1])
    assert float(converted[0]) == pytest.approx(float(explicit[0]), rel=1e-6)


def test_a_label_count_mismatch_is_refused():
    """Student and teacher may score different prompts (OPSD), but must share the response tokens."""
    teacher = TeacherOutput(full_logits=TEACHER, labels=torch.tensor([[1, 2, 3]]))
    with pytest.raises(AssertionError, match='Label count mismatch'):
        extract_active(STUDENT, teacher, torch.tensor([[1, -100, -100]]))


def test_a_teacher_output_with_neither_representation_is_refused():
    with pytest.raises(ValueError, match='full_logits'):
        TeacherOutput().validate()


def test_a_teacher_without_labels_falls_back_to_the_student_mask():
    """A pipeline-parallel non-last stage carries a placeholder ``TeacherOutput`` with no labels; the
    student's mask is then the only one available and must be used as-is."""
    teacher = TeacherOutput(full_logits=TEACHER, labels=None)
    student_active, teacher_active, num_valid = extract_active(STUDENT, teacher, LABELS)
    assert int(num_valid) == 2
    assert student_active.shape == TEACHER[ACTIVE].shape
    assert teacher_active.full_logits.shape == TEACHER[ACTIVE].shape


def test_to_device_moves_every_tensor_field():
    teacher = _topk_teacher()
    assert teacher.to_device(torch.device('cpu')) is teacher
    assert teacher.topk_logprobs.device.type == 'cpu'
    assert teacher.labels.device.type == 'cpu'


def test_to_topk_on_an_already_topk_teacher_is_a_no_op():
    teacher = _topk_teacher()
    assert teacher.to_topk(2) is teacher


def test_vocab_alignment_copies_the_wider_side_into_the_narrower():
    """The extra vocabulary entries must carry the same logits on both sides, or the JSD would be
    computed over two different supports."""
    student, teacher = _align_vocab(torch.tensor([[1., 2.]]), torch.tensor([[3., 4., 5.]]))
    assert student.tolist() == [[1., 2., 5.]]  # the narrower student borrows the teacher's extra logits
    assert teacher.tolist() == [[3., 4., 5.]]

    student, teacher = _align_vocab(torch.tensor([[1., 2., 3.]]), torch.tensor([[4., 5.]]))
    assert student.tolist() == [[1., 2., 3.]]
    assert teacher.tolist() == [[4., 5., 3.]]  # the narrower teacher borrows the student's extra logits


def test_topk_mode_is_recognised_by_the_teacher_output():
    assert TeacherOutput(full_logits=TEACHER).is_topk_mode is False
    assert _topk_teacher().is_topk_mode is True

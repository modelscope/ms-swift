import math
import torch
from collections import defaultdict, deque
from types import SimpleNamespace

from swift.rl_core.data import GKDSample
from swift.rlhf_trainers.gkd_helpers import assemble_teacher_output
from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_monitoring_stats
from swift.rlhf_trainers.gkd_trainer import GKDTrainer
from swift.rlhf_trainers.utils import parse_prompt_logprobs


def test_gkd_monitoring_full_logits_overlap_and_exact_gap():
    student = torch.tensor([[[-9., -9., -9., -9.], [4., 3., 1., 0.], [4., 0., 3., 2.]]])
    teacher = torch.tensor([[[-9., -9., -9., -9.], [4., 3., 0., 1.], [0., 1., 4., 3.]]])
    labels = torch.tensor([[-100, 1, 2]])
    teacher_out = TeacherOutput(full_logits=teacher, labels=labels)

    stats = gkd_monitoring_stats(student, teacher_out, labels, full_vocab_topk=2)
    assert torch.isclose(stats['topk_overlap_sum'] / stats['topk_overlap_count'], torch.tensor(0.75))

    expected = (torch.log_softmax(teacher[0, 1], -1)[1] - torch.log_softmax(student[0, 1], -1)[1]
                + torch.log_softmax(teacher[0, 2], -1)[2] - torch.log_softmax(student[0, 2], -1)[2]) / 2
    actual = stats['teacher_student_gap_sum'] / stats['teacher_student_gap_count']
    torch.testing.assert_close(actual, expected)


def test_gkd_monitoring_topk_uses_observed_token_logprob_outside_topk():
    student = torch.tensor([[[3., 2., 1., 0.]]])
    labels = torch.tensor([[3]])  # observed token is outside the teacher's retained top-2
    teacher_target_lp = torch.tensor([[-0.25]])
    teacher_out = TeacherOutput(
        topk_logprobs=torch.tensor([[[-0.1, -0.2]]]),
        topk_indices=torch.tensor([[[0, 1]]]),
        target_logprobs=teacher_target_lp,
        labels=labels,
    )

    stats = gkd_monitoring_stats(student, teacher_out, labels)
    expected_gap = teacher_target_lp.item() - torch.log_softmax(student[0, 0], -1)[3].item()
    assert stats['teacher_student_gap_count'].item() == 1
    assert math.isclose(
        (stats['teacher_student_gap_sum'] / stats['teacher_student_gap_count']).item(), expected_gap, rel_tol=1e-6)


def test_teacher_api_keeps_observed_token_outside_topk_for_gap():
    response = SimpleNamespace(prompt_logprobs=[
        None,
        {
            0: {
                'logprob': -0.1
            },
            1: {
                'logprob': -0.2
            },
            9: {
                'logprob': -2.5
            },
        },
        {
            2: {
                'logprob': -0.3
            },
            0: {
                'logprob': -0.4
            },
            3: {
                'logprob': -3.5
            },
        },
    ])
    parsed = [parse_prompt_logprobs(response, topk=2, include_sampled=True)]
    inputs = {
        'input_ids': torch.tensor([[5, 9, 3]]),
        'labels': torch.tensor([[-100, 9, 3]]),
        'attention_mask': torch.ones(1, 3, dtype=torch.long),
    }

    teacher_out = assemble_teacher_output(parsed, inputs, topk=2, template_padding_free=False, device='cpu')

    torch.testing.assert_close(teacher_out.target_logprobs[0, :2], torch.tensor([-2.5, -3.5]))
    assert teacher_out.topk_indices[0, 0].tolist() == [0, 1]
    assert teacher_out.topk_indices[0, 1].tolist() == [2, 0]


def test_gkd_rollout_logs_num_turns():
    trainer = GKDTrainer.__new__(GKDTrainer)
    trainer.model = SimpleNamespace(training=True)
    trainer.log_completions = True
    trainer._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
    trainer._logs = {'prompt': deque(), 'completion': deque()}
    trainer._gather_and_flatten = lambda values, **_: values
    samples = [
        GKDSample(messages=[], rollout_infos={'num_turns': 2}),
        GKDSample(messages=[], rollout_infos={'num_turns': 4}),
    ]

    trainer._log_rollout(samples)

    assert trainer._metrics['train']['num_turns'] == [3.0]
    assert list(trainer._logs['num_turns']) == [2, 4]

import copy
import torch
from datasets import Dataset

from swift.dataset.preprocessor.core import RowPreprocessor
from swift.rl_core.data import GKDSample
from swift.rlhf_trainers.gkd_helpers import build_teacher_requests, encode_gkd_samples
from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss
from swift.template.template_inputs import StdTemplateInputs


class _FakeTemplate:
    """Minimal template that makes media-dependent prompt lengths observable."""

    enable_thinking = None

    def encode(self, row, return_length=False):
        inputs = StdTemplateInputs.from_dict(row)
        images = list(inputs.images or [])
        assistant = next(message for message in reversed(inputs.messages) if message['role'] == 'assistant')
        content = assistant['content']
        response_ids = list(content.get('token_ids', [])) if isinstance(content, dict) else list(content)
        prompt_length = 2 + 2 * len(images)
        return {
            'input_ids': [0] * prompt_length + response_ids,
            'labels': [-100] * prompt_length + response_ids,
            'pixel_values': images,
            'length': prompt_length + len(response_ids),
        }


def _sample(*, student_images, teacher_images):
    teacher_image_count = len(student_images) if teacher_images is None else len(teacher_images)
    return GKDSample(
        messages=[
            {
                'role': 'user',
                'content': '<image>' * len(student_images) + '\nquestion'
            },
            {
                'role': 'assistant',
                'content': 'rollout'
            },
        ],
        images=student_images,
        teacher_prompt='<image>' * teacher_image_count + '\nprivileged question',
        teacher_images=teacher_images,
        response_token_ids=[[31, 32, 33]],
    )


def test_teacher_images_survive_preprocessing_and_sample_roundtrip():
    row = {
        'messages': [{
            'role': 'user',
            'content': '<image>question'
        }],
        'images': ['student.jpg'],
        'teacher_prompt': '<image>privileged question',
        'teacher_images': ['teacher.jpg'],
    }
    RowPreprocessor._cast_mm_data(row)

    assert 'teacher_images' in RowPreprocessor.standard_keys
    assert row['images'] == [{'bytes': None, 'path': 'student.jpg'}]
    assert row['teacher_images'] == [{'bytes': None, 'path': 'teacher.jpg'}]

    sample = GKDSample.from_row(row)
    cloned = copy.deepcopy(sample)
    assert cloned.teacher_images == [{'bytes': None, 'path': 'teacher.jpg'}]
    assert 'teacher_images' not in cloned.extra


def test_dataset_preprocessor_preserves_teacher_images_column():
    dataset = Dataset.from_dict({
        'messages': [[{
            'role': 'user',
            'content': '<image>question'
        }]],
        'images': [['student.jpg']],
        'teacher_prompt': ['<image>privileged question'],
        'teacher_images': [['teacher.jpg']],
    })

    processed = RowPreprocessor()(dataset, load_from_cache_file=False, strict=True)

    assert processed[0]['images'] == [{'bytes': None, 'path': 'student.jpg'}]
    assert processed[0]['teacher_images'] == [{'bytes': None, 'path': 'teacher.jpg'}]


def test_teacher_images_fall_back_to_student_images():
    sample = _sample(student_images=['student-1', 'student-2'], teacher_images=None)
    student_rows, teacher_rows, has_opsd = encode_gkd_samples([sample], _FakeTemplate())

    assert has_opsd
    assert teacher_rows[0]['pixel_values'] == student_rows[0]['pixel_values']


def test_explicit_empty_teacher_images_builds_text_only_teacher_view():
    sample = _sample(student_images=['student-1', 'student-2'], teacher_images=[])
    _, teacher_rows, has_opsd = encode_gkd_samples([sample], _FakeTemplate())

    assert has_opsd
    assert teacher_rows[0]['pixel_values'] == []


def test_different_image_counts_reencode_prompt_and_align_response():
    sample = _sample(
        student_images=[f'student-{i}' for i in range(8)],
        teacher_images=['teacher-1', 'teacher-2'],
    )
    student_rows, teacher_rows, has_opsd = encode_gkd_samples([sample], _FakeTemplate())

    assert has_opsd
    assert student_rows[0]['pixel_values'] == [f'student-{i}' for i in range(8)]
    assert teacher_rows[0]['pixel_values'] == ['teacher-1', 'teacher-2']
    assert len(student_rows[0]['input_ids']) != len(teacher_rows[0]['input_ids'])
    student_response = [token for token in student_rows[0]['labels'] if token != -100]
    teacher_response = [token for token in teacher_rows[0]['labels'] if token != -100]
    assert student_response == teacher_response == [31, 32, 33]


def test_teacher_request_uses_teacher_images():
    sample = _sample(
        student_images=['student-1', 'student-2'],
        teacher_images=['teacher-1'],
    )
    assert sample.build_teacher_view()

    request = build_teacher_requests([sample])[0]
    assert request.images == ['teacher-1']
    assert request.messages == sample.teacher_messages


def test_full_vocab_jsd_backpropagates_only_through_student():
    student_logits = torch.nn.Parameter(torch.randn(1, 6, 11))
    teacher_source = torch.nn.Parameter(torch.randn(1, 4, 11))
    student_labels = torch.tensor([[-100, -100, -100, 4, 5, 6]])
    teacher_labels = torch.tensor([[-100, 4, 5, 6]])
    teacher_output = TeacherOutput(full_logits=teacher_source.detach(), labels=teacher_labels)

    loss_total, count = gkd_loss(student_logits, teacher_output, student_labels, beta=0.5, temperature=1.0)
    loss = loss_total / count
    loss.backward()

    assert torch.isfinite(loss)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert teacher_source.grad is None

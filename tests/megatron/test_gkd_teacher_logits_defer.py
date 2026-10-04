# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace
from unittest import mock


def _import_trainer():
    try:
        from swift.megatron.trainers.gkd_trainer import MegatronGKDTrainer
    except Exception as e:  # noqa: megatron-core not installed in this env
        print(f'SKIP gkd teacher defer tests: {e}')
        return None
    return MegatronGKDTrainer


def test_compute_teacher_logits_defers_full_vocab_fixed_teacher():
    """Full-vocabulary local teacher logits must not be materialized at batch prep."""
    trainer_cls = _import_trainer()
    if trainer_cls is None:
        return

    stub = SimpleNamespace(
        use_teacher_api=False,
        _is_self_distillation=False,
        gkd_logits_topk=None,
    )
    encoded_batches = [{
        'teacher_model_inputs': {
            'input_ids': object(),
        },
    }]
    with mock.patch.object(trainer_cls, '_compute_teacher_logits_local') as local_mock:
        trainer_cls._compute_teacher_logits(stub, encoded_batches)
        local_mock.assert_not_called()
    assert 'teacher_output' not in encoded_batches[0]


def test_compute_teacher_logits_eager_when_topk_configured():
    """Compressed top-k teacher logits remain eager at batch preparation."""
    trainer_cls = _import_trainer()
    if trainer_cls is None:
        return

    stub = SimpleNamespace(
        use_teacher_api=False,
        _is_self_distillation=False,
        gkd_logits_topk=8,
    )
    encoded_batches = [{'teacher_model_inputs': {}}]
    with mock.patch.object(trainer_cls, '_compute_teacher_logits_local') as local_mock:
        trainer_cls._compute_teacher_logits(stub, encoded_batches)
        local_mock.assert_called_once_with(stub, encoded_batches, None)


def test_compute_teacher_output_local_used_by_forward_step_when_missing():
    """forward_step computes teacher logits just-in-time for full-vocabulary mode."""
    trainer_cls = _import_trainer()
    if trainer_cls is None:
        return

    try:
        import torch

        from swift.rlhf_trainers.gkd_loss import TeacherOutput
    except Exception as e:
        print(f'SKIP forward_step defer test: {e}')
        return

    teacher_out = TeacherOutput(full_logits=torch.zeros(1, 2, 4))
    stub = SimpleNamespace(
        gkd_logits_topk=None,
        _prepare_batch=lambda batch, vp_stage: batch,
        _compute_teacher_output_local=mock.Mock(return_value=teacher_out),
        loss_func=mock.Mock(),
    )
    data_iterator = iter([{
        'data_source': 'dataset',
        'input_ids': torch.zeros((1, 2), dtype=torch.long),
        'teacher_model_inputs': {
            'input_ids': torch.zeros((1, 2), dtype=torch.long),
        },
    }])
    model = mock.Mock()
    model.return_value = torch.zeros(1, 2, 4)
    unwrapped = mock.Mock()
    unwrapped.get_input_tensor.return_value = None
    unwrapped.vp_stage = None
    with mock.patch('swift.megatron.trainers.gkd_trainer.get_attr_wrapped_model', return_value=unwrapped):
        trainer_cls.forward_step(stub, data_iterator, model)

    stub._compute_teacher_output_local.assert_called_once()
    model.assert_called_once()

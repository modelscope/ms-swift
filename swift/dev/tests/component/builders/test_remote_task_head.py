# Copyright (c) ModelScope Contributors. All rights reserved.
"""Guards for the remote (server) training of task heads on a plain causal-lm host.

``mode='remote'`` refused the task types outright until the refusal was lifted: the loss is set
by CLASS NAME (instances cannot cross HTTP) and the readout head rides the adapter channel
(``add_adapter_to_model(task=...)``). These tests pin the dispatch -- each task's loss class and
constructor kwargs must match what the local ``configure_*_loss`` family would build, or the
remote run trains a wrong objective while the local reference trains the right one.
"""
from typing import Any, Dict, List, Optional

import pytest

from swift.dev.builders.remote import _remote_task_head_kwargs, configure_remote_loss
from swift.dev.config import ModelConfig, TrainConfig


class _RecordingModel:
    """The client-side handle face: names + scalar kwargs only."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def set_loss(self, loss_cls: str, **kwargs: Any) -> None:
        self.calls.append(('set_loss', loss_cls, kwargs))


def _model(task_type: str, num_labels: Optional[int] = None, problem_type: Optional[str] = None) -> ModelConfig:
    return ModelConfig(model='m', task_type=task_type, num_labels=num_labels, problem_type=problem_type)


def test_embedding_dispatches_the_contrastive_loss_by_class_name():
    model = _RecordingModel()
    configure_remote_loss(model, TrainConfig(), task_type='embedding', model_config=_model('embedding'))
    assert model.calls == [('set_loss', 'InfonceLoss', {})]


def test_seq_cls_dispatches_with_problem_type_and_num_labels():
    model = _RecordingModel()
    mc = _model('seq_cls', num_labels=3, problem_type='single_label_classification')
    configure_remote_loss(model, TrainConfig(), task_type='seq_cls', model_config=mc)
    assert model.calls == [
        ('set_loss', 'SeqClsLoss', {'problem_type': 'single_label_classification', 'num_labels': 3})]


def test_seq_cls_without_num_labels_fails_loudly():
    model = _RecordingModel()
    mc = _model('seq_cls', problem_type='single_label_classification')
    with pytest.raises(ValueError, match='num_labels is required'):
        configure_remote_loss(model, TrainConfig(), task_type='seq_cls', model_config=mc)


def test_reranker_and_generative_reranker_dispatch_the_pointwise_loss():
    for task_type in ('reranker', 'generative_reranker'):
        model = _RecordingModel()
        configure_remote_loss(model, TrainConfig(), task_type=task_type, model_config=_model(task_type))
        assert model.calls == [('set_loss', 'PointwiseRerankerLoss', {})], task_type


def test_decision_dispatches_the_dev_scoring_loss_by_class_name():
    model = _RecordingModel()
    configure_remote_loss(model, TrainConfig(), task_type='decision', model_config=_model('decision'))
    assert model.calls == [('set_loss', 'ScoringLoss', {})]


def test_causal_lm_main_line_is_unchanged():
    model = _RecordingModel()
    configure_remote_loss(model, TrainConfig(), task_type=None, model_config=_model('causal_lm'))
    name, kwargs = model.calls[0][1], model.calls[0][2]
    assert name == 'CrossEntropyLoss'
    assert kwargs['reduction'] == 'sum' and kwargs['label_smoothing'] == 0.0


def test_head_kwargs_build_the_server_side_head():
    assert _remote_task_head_kwargs('embedding', _model('embedding')) == {}
    assert _remote_task_head_kwargs('generative_reranker', _model('generative_reranker')) == {}
    assert _remote_task_head_kwargs('seq_cls', _model('seq_cls', num_labels=3)) == {
        'task': 'seq_cls', 'num_labels': 3}
    assert _remote_task_head_kwargs('reranker', _model('reranker')) == {'task': 'reranker', 'num_labels': 1}


def test_head_kwargs_without_num_labels_fail_loudly():
    with pytest.raises(ValueError, match='num_labels is required'):
        _remote_task_head_kwargs('seq_cls', _model('seq_cls'))


def test_register_loss_exports_dev_classes_into_the_twinkle_namespace():
    # The decision losses live in swift.dev; set_loss resolves a class NAME through
    # twinkle.loss (construct_class), so register_loss must have exported them there.
    import twinkle.loss
    from twinkle.loss import Loss
    from twinkle.utils import construct_class

    import swift.dev.loss.decision  # noqa: F401 -- register_loss side effect

    for cls_name in ('ScoringLoss', 'DistillRPSLoss', 'SmoothBrierLoss', 'CERPSLoss'):
        instance = construct_class(cls_name, Loss, twinkle.loss)
        assert type(instance).__name__ == cls_name

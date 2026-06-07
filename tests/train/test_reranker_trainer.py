from types import SimpleNamespace

import torch

from swift.trainers.reranker_trainer import RerankerTrainer, gather_for_reranker_metrics
from swift.trainers.trainer import Trainer


def _fake_trainer_init(self, *args, **kwargs):
    self.args = kwargs.get('args', SimpleNamespace(loss_type=None))
    self.label_names = ['labels']
    self.gather_function = None


def test_pointwise_reranker_adds_group_sizes_to_label_names(monkeypatch):
    monkeypatch.setattr(Trainer, '__init__', _fake_trainer_init)

    trainer = RerankerTrainer(args=SimpleNamespace(loss_type='pointwise_reranker'))

    assert trainer.label_names == ['labels', 'group_sizes']
    assert trainer.gather_function is gather_for_reranker_metrics


def test_listwise_reranker_keeps_default_label_names(monkeypatch):
    monkeypatch.setattr(Trainer, '__init__', _fake_trainer_init)

    trainer = RerankerTrainer(args=SimpleNamespace(loss_type='listwise_reranker'))

    assert trainer.label_names == ['labels']
    assert trainer.gather_function is gather_for_reranker_metrics


def test_gather_for_reranker_metrics_preserves_tuple_labels():
    labels = torch.tensor([1, 0, 0, 1], dtype=torch.long)
    group_sizes = torch.tensor([2, 2], dtype=torch.long)

    gathered = gather_for_reranker_metrics((labels, group_sizes))

    assert isinstance(gathered, tuple)
    assert torch.equal(gathered[0], labels)
    assert torch.equal(gathered[1], group_sizes)


def test_evaluation_loop_preserves_metric_prefix(monkeypatch):
    monkeypatch.setattr(Trainer, '__init__', _fake_trainer_init)

    captured = {}

    def _fake_evaluation_loop(self, *args, **kwargs):
        captured['metric_key_prefix'] = kwargs.get('metric_key_prefix')
        self.gather_function = object()
        return SimpleNamespace(metrics={'test_mrr': 0.75})

    monkeypatch.setattr(Trainer, 'evaluation_loop', _fake_evaluation_loop)

    trainer = RerankerTrainer(args=SimpleNamespace(loss_type='pointwise_reranker'))
    output = trainer.evaluation_loop(metric_key_prefix='test')

    assert captured['metric_key_prefix'] == 'test'
    assert output.metrics == {'test_mrr': 0.75}
    assert trainer.gather_function is gather_for_reranker_metrics

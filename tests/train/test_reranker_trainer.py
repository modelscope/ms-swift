from types import SimpleNamespace

from swift.trainers.reranker_trainer import RerankerTrainer
from swift.trainers.trainer import Trainer
from swift.trainers.utils import gather_for_unpadded_tensors


def _fake_trainer_init(self, *args, **kwargs):
    self.args = kwargs.get('args', SimpleNamespace(loss_type=None))
    self.label_names = ['labels']
    self.gather_function = None


def test_pointwise_reranker_adds_group_sizes_to_label_names(monkeypatch):
    monkeypatch.setattr(Trainer, '__init__', _fake_trainer_init)

    trainer = RerankerTrainer(args=SimpleNamespace(loss_type='pointwise_reranker'))

    assert trainer.label_names == ['labels', 'group_sizes']
    assert trainer.gather_function is gather_for_unpadded_tensors


def test_listwise_reranker_keeps_default_label_names(monkeypatch):
    monkeypatch.setattr(Trainer, '__init__', _fake_trainer_init)

    trainer = RerankerTrainer(args=SimpleNamespace(loss_type='listwise_reranker'))

    assert trainer.label_names == ['labels']
    assert trainer.gather_function is gather_for_unpadded_tensors


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
    assert trainer.gather_function is gather_for_unpadded_tensors

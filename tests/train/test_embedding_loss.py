# Copyright (c) ModelScope Contributors. All rights reserved.
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from swift.loss.embedding import InfonceLoss, _parse_multi_negative_sentences
from swift.metrics import MeanMetric
from swift.trainers.embedding_trainer import EmbeddingTrainer, Trainer


@pytest.mark.parametrize('negative_count', [1, 2])
def test_infonce_padding_samples_only_negatives(monkeypatch, negative_count):
    # Cycle through the complete sampling population, including its last entry.
    monkeypatch.setattr(np.random, 'choice', lambda a, size, replace: np.resize(a, size))
    groups = torch.arange(2 * (negative_count + 2) * 3).reshape(2, negative_count + 2, 3)
    labels = torch.tensor([1] + [0] * negative_count).repeat(2)

    actual = _parse_multi_negative_sentences(groups.flatten(0, 1), labels, hard_negatives=3 * negative_count)

    for original, padded in zip(groups, actual):
        expected = torch.cat((original, original[2:].repeat(2, 1)))
        torch.testing.assert_close(padded, expected)


@pytest.mark.parametrize('hard_negatives', [None, 2, 3])
def test_infonce_without_padding_preserves_samples(hard_negatives):
    groups = torch.arange(30).reshape(2, 5, 3)
    labels = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0])

    actual = _parse_multi_negative_sentences(groups.flatten(0, 1), labels, hard_negatives)

    end = 5 if hard_negatives is None else hard_negatives + 2
    for original, parsed in zip(groups, actual):
        torch.testing.assert_close(parsed, original[:end])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('use_batch', [False, True])
def test_infonce_padded_loss_and_gradients(monkeypatch, device, dtype, use_batch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    for name, value in {
            'RANK': '0',
            'WORLD_SIZE': '1',
            'INFONCE_TEMPERATURE': '0.5',
            'INFONCE_HARD_NEGATIVES': '3',
            'INFONCE_USE_BATCH': str(use_batch),
            'INFONCE_MASK_FAKE_NEGATIVE': 'False',
            'INFONCE_INCLUDE_QQ': 'False',
            'INFONCE_INCLUDE_DD': 'False',
    }.items():
        monkeypatch.setenv(name, value)
    embeddings = F.normalize(
        torch.tensor([[1., 0., 0.], [1., 1., 0.], [-1., 0., 1.], [0., 1., 0.], [0., 1., 1.], [1., -1., 0.]],
                     device=device,
                     dtype=dtype),
        dim=-1).requires_grad_()
    labels = torch.tensor([1, 0, 1, 0], device=device)
    actual = InfonceLoss(None, None)({'last_hidden_state': embeddings}, labels)

    # Each example has one negative, so padding must repeat that negative twice.
    reference = embeddings.detach().clone().requires_grad_()
    queries = reference[[0, 3]]
    documents = reference[[1, 2, 2, 2, 4, 5, 5, 5]]
    if use_batch:
        logits = queries @ documents.T / 0.5
        targets = torch.tensor([0, 4], device=device)
    else:
        logits = (queries[:, None] * documents.reshape(2, 4, 3)).sum(-1) / 0.5
        targets = torch.zeros(2, dtype=torch.long, device=device)
    expected = F.cross_entropy(logits, targets)

    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, embeddings)
    expected_grad, = torch.autograd.grad(expected, reference)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_mrl_loss_logs_each_dimension(monkeypatch):
    monkeypatch.setattr(Trainer, '__init__', lambda self, *args, **kwargs: None)
    trainer = EmbeddingTrainer.__new__(EmbeddingTrainer)
    trainer.args = SimpleNamespace(mrl_dims={2: 0.25, 4: 0.75, 8: 1.0})
    trainer.model = SimpleNamespace(training=True)
    trainer.custom_metrics = {
        mode: defaultdict(lambda: MeanMetric(nan_value=None, device=torch.device('cpu')))
        for mode in ('train', 'eval')
    }

    def loss_func(outputs, labels, **kwargs):
        return outputs['last_hidden_state'].square().sum() + labels.float().sum()

    trainer.compute_loss_func = loss_func
    EmbeddingTrainer.__init__(trainer)

    hidden_states = torch.tensor([[3., 4., 0., 0.], [0., 0., 5., 12.]], requires_grad=True)
    labels = torch.tensor([1, 0])
    loss = trainer.compute_loss_func({'last_hidden_state': hidden_states}, labels)

    dim_2_loss = loss_func({'last_hidden_state': F.normalize(hidden_states[..., :2], p=2, dim=-1)}, labels)
    dim_4_loss = loss_func({'last_hidden_state': F.normalize(hidden_states, p=2, dim=-1)}, labels)
    torch.testing.assert_close(loss, 0.25 * dim_2_loss + 0.75 * dim_4_loss)
    assert trainer.custom_metrics['train']['mrl_loss_2'].compute()['value'] == pytest.approx(dim_2_loss.item())
    assert trainer.custom_metrics['train']['mrl_loss_4'].compute()['value'] == pytest.approx(dim_4_loss.item())
    assert 'mrl_loss_8' not in trainer.custom_metrics['train']

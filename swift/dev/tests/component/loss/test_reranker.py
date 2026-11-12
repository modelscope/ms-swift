# Copyright (c) ModelScope Contributors. All rights reserved.
"""The reranker losses behind ``--loss_type pointwise_reranker|listwise_reranker``.

``PointwiseRerankerLoss`` is a plain BCE over the squeezed score, asserted against the same
``binary_cross_entropy_with_logits`` the implementation uses. ``ListwiseRerankerLoss`` recovers query
groups from the label vector (a 1 starts a group) and treats ranking as picking the positive out of
the group -- so the group boundaries, the minimum-size skip rule, and the temperature all change what
the denominator is, and each is pinned.
"""
import types

import pytest
import torch
import torch.nn.functional as F

from swift.loss.reranker import ListwiseRerankerLoss, PointwiseRerankerLoss


class _Trainer:
    pass


def _loss(cls):
    return cls(args=None, trainer=_Trainer())


def _outputs(*scores):
    return types.SimpleNamespace(logits=torch.tensor([[score] for score in scores], dtype=torch.float))


def test_pointwise_loss_is_bce_over_the_squeezed_score():
    outputs = types.SimpleNamespace(logits=torch.tensor([[2.], [-1.]]))
    loss = _loss(PointwiseRerankerLoss)(outputs, torch.tensor([1, 0]))
    expected = F.binary_cross_entropy_with_logits(torch.tensor([2., -1.]), torch.tensor([1., 0.]))
    assert float(loss) == pytest.approx(float(expected))


def test_pointwise_loss_accepts_integer_labels():
    loss = _loss(PointwiseRerankerLoss)(_outputs(2., -1.), torch.tensor([1, 0]))
    assert float(loss) == pytest.approx(0.22009489, rel=1e-6)


def test_listwise_loss_averages_cross_entropy_over_query_groups():
    loss = _loss(ListwiseRerankerLoss)(_outputs(3., 2., 1., 0.5, 0.2, 0.1), torch.tensor([1, 0, 0, 1, 0, 0]))
    group1 = F.cross_entropy(torch.tensor([[3., 2., 1.]]), torch.tensor([0]))
    group2 = F.cross_entropy(torch.tensor([[0.5, 0.2, 0.1]]), torch.tensor([0]))
    assert float(loss) == pytest.approx(float((group1 + group2) / 2))


def test_a_group_smaller_than_the_minimum_is_skipped_not_scored(monkeypatch):
    """A 3-candidate group and a 2-candidate group: raising the minimum to 3 drops the small group
    out of the mean instead of letting it drag the average down."""
    outputs = _outputs(3., 2., 1., 0.5, 0.2)
    labels = torch.tensor([1, 0, 0, 1, 0])
    monkeypatch.setenv('LISTWISE_RERANKER_MIN_GROUP_SIZE', '3')
    loss = _loss(ListwiseRerankerLoss)(outputs, labels)
    group1 = F.cross_entropy(torch.tensor([[3., 2., 1.]]), torch.tensor([0]))
    assert float(loss) == pytest.approx(float(group1))


def test_labels_without_a_positive_score_zero():
    """The zero is built with requires_grad=True on purpose: the Trainer calls backward() on it."""
    loss = _loss(ListwiseRerankerLoss)(_outputs(3., 2., 1.), torch.tensor([0, 0, 0]))
    assert loss.requires_grad
    assert float(loss.detach()) == 0.0


def test_groups_of_one_are_all_skipped_so_the_loss_is_zero():
    loss = _loss(ListwiseRerankerLoss)(_outputs(3., 1.), torch.tensor([1, 1]))
    assert float(loss.detach()) == 0.0


def test_temperature_scales_the_logits_before_the_softmax(monkeypatch):
    monkeypatch.setenv('LISTWISE_RERANKER_TEMPERATURE', '2.0')
    outputs = _outputs(3., 2., 1., 0.5, 0.2)
    labels = torch.tensor([1, 0, 0, 1, 0])
    loss = _loss(ListwiseRerankerLoss)(outputs, labels)
    group1 = F.cross_entropy(torch.tensor([[1.5, 1.0, 0.5]]), torch.tensor([0]))
    group2 = F.cross_entropy(torch.tensor([[0.25, 0.1]]), torch.tensor([0]))
    assert float(loss) == pytest.approx(float((group1 + group2) / 2))

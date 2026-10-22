# Copyright (c) ModelScope Contributors. All rights reserved.
"""``RerankerMetrics``: MRR and NDCG for ``--eval_metric reranker``.

The metric is defined on *queries*, and the query boundaries are not passed in -- they are recovered
from the labels, where a positive (``1``) starts a new query and the negatives (``0``) after it belong
to it. Everything here follows from that: a query is ranked in isolation, a query with nothing to
retrieve against is skipped rather than scored 0, and the reported number is the mean over the queries
that could actually be scored. Getting the grouping wrong silently changes the denominator, so each
boundary case is pinned.
"""
import numpy as np
import pytest
import torch
from transformers.trainer_utils import EvalPrediction

from swift.metrics.reranker import RerankerMetrics


def _reranker():
    return RerankerMetrics(args=None, trainer=None)


def _eval_prediction(predictions, label_ids):
    return EvalPrediction(predictions=np.asarray(predictions), label_ids=np.asarray(label_ids))


def test_a_query_with_its_positive_on_top_scores_one():
    """Two queries, each with the positive ranked first: perfect MRR and perfect NDCG."""
    result = _reranker()._calculate_metrics(np.array([0.9, 0.1, 0.8, 0.3]), np.array([1, 0, 1, 0]))
    assert result['mrr'] == pytest.approx(1.0)
    assert result['ndcg'] == pytest.approx(1.0)


def test_torch_tensors_are_scored_the_same_as_numpy():
    """``_calculate_metrics`` also receives torch tensors (model output), and must flatten and score
    them identically."""
    result = _reranker()._calculate_metrics(torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3]), torch.tensor([1, 0, 0, 1, 0]))
    assert result['mrr'] == pytest.approx(2 / 3)
    assert result['ndcg'] == pytest.approx(0.75)


def test_a_positive_ranked_third_scores_a_third():
    """The first query's positive is last of three, the second's is first: the mean is (1/3 + 1) / 2.

    NDCG discounts rather than reciprocates, so the same ranking gives (0.5 + 1) / 2.
    """
    result = _reranker()._calculate_metrics(
        np.array([0.1, 0.9, 0.2, 0.8, 0.3]), np.array([1, 0, 0, 1, 0]))
    assert result['mrr'] == pytest.approx(2 / 3)
    assert result['ndcg'] == pytest.approx(0.75)


def test_a_query_boundary_is_the_next_positive_not_a_fixed_width():
    """Grouping is by positive position, so an uneven batch is handled without being reshaped."""
    result = _reranker()._calculate_metrics(
        np.array([0.1, 0.9, 0.2, 0.8, 0.3]), np.array([1, 0, 0, 1, 0]))
    assert result['mrr'] == pytest.approx(2 / 3)
    # A width-2 grouping would have split [0.1, 0.9] / [0.2, 0.8] / [0.3] instead.
    assert result['ndcg'] != pytest.approx(1.0)


def test_labels_without_any_positive_report_zero_rather_than_crashing():
    result = _reranker()._calculate_metrics(np.array([0.1, 0.2]), np.array([0, 0]))
    assert result == {'mrr': 0.0, 'ndcg': 0.0}


def test_a_query_with_a_single_candidate_is_skipped():
    """One item is nothing to rank, so it contributes no observation and the result is empty."""
    result = _reranker()._calculate_metrics(np.array([0.1]), np.array([1]))
    assert result == {'mrr': 0.0, 'ndcg': 0.0}


def test_a_degenerate_query_is_skipped_while_the_others_still_count():
    """A trailing one-item query must not be averaged in as a zero -- the mean stays over two queries."""
    result = _reranker()._calculate_metrics(
        np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.5]), np.array([1, 0, 0, 1, 0, 1]))
    assert result['mrr'] == pytest.approx(2 / 3)
    assert result['ndcg'] == pytest.approx(0.75)


def test_update_accumulates_batches_and_compute_averages_over_all_queries():
    """``update`` is per batch, so the reported number must cover every query seen so far, not the last
    batch."""
    metric = _reranker()
    metric.update(torch.tensor([0.9, 0.1, 0.8, 0.3]), torch.tensor([1, 0, 1, 0]))
    metric.update(torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3]), torch.tensor([1, 0, 0, 1, 0]))
    result = metric.compute()
    assert result['mrr'] == pytest.approx(0.8333333333333333)
    assert result['ndcg'] == pytest.approx(0.875)


def test_reset_clears_the_accumulated_batches():
    metric = _reranker()
    metric.update(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
    metric.reset()
    assert metric.logits == []
    assert metric.labels == []


def test_compute_metrics_reads_an_eval_prediction_directly():
    """The HF hook path: the whole validation set arrives at once, without ``update``."""
    metrics = _reranker().compute_metrics(_eval_prediction([0.1, 0.9, 0.2, 0.8, 0.3], [1, 0, 0, 1, 0]))
    assert metrics['mrr'] == pytest.approx(2 / 3)
    assert metrics['ndcg'] == pytest.approx(0.75)

# Copyright (c) ModelScope Contributors. All rights reserved.
"""``InfonceMetrics`` and ``PairedMetrics``: the retrieval metrics behind ``--eval_metric infonce|paired``.

Both are defined on *queries* rather than on rows. For infonce a query is an anchor plus its positive
plus its negatives, and the query boundaries are recovered from the label vector instead of being
passed in -- so the label length is part of the contract and is pinned below. ``INFONCE_USE_BATCH``
then decides whether the other queries' positives also count as negatives, which is the difference
between a margin of 0.7 and one of 0.0 on the same data; both readings are asserted.

``PairedMetrics`` needs scikit-learn for its paired distance functions, and scikit-learn is not in
``requirements/``, so those cases skip when it is absent rather than failing the suite.
"""
import numpy as np
import pytest
from transformers.trainer_utils import EvalPrediction

from swift.metrics.embedding import EmbedddingMetricMixin, InfonceMetrics


class _Args:
    pass


def _infonce():
    return InfonceMetrics(_Args(), object())


# One query: anchor, positive, then two negatives. Two queries: the same twice over.
_ONE_QUERY = np.array([[1., 0], [0.8, 0.6], [0, 1], [0, 0]])
_TWO_QUERIES = np.array([[1., 0], [0.8, 0.6], [0, 1], [1., 0], [0.6, 0.8], [0, 1]])
# The label vector is one entry shorter per query than the row count -- see the test below.
_ONE_QUERY_LABELS = np.array([1, 0, 0, 0])
_TWO_QUERY_LABELS = np.array([1, 0, 1, 0])


def test_a_margin_is_positive_when_the_positive_scores_highest():
    result = _infonce()._calculate_metrics(_ONE_QUERY, _ONE_QUERY_LABELS)
    assert result['mean_pos'] == pytest.approx(0.8)
    assert result['mean_neg'] == pytest.approx(0.0)
    assert result['margin'] == pytest.approx(0.8)


def test_a_margin_goes_negative_when_a_negative_outscores_the_positive():
    """The margin is the signal this metric exists for: a positive below the hardest negative is a
    retrieval failure, and must not be hidden by averaging it against the easy negatives."""
    rows = np.array([[1., 0], [0.6, 0.8], [0.9, 0.4358898943540674], [0, 1]])
    result = _infonce()._calculate_metrics(rows, _ONE_QUERY_LABELS)
    assert result['mean_pos'] == pytest.approx(0.6)
    assert result['mean_neg'] == pytest.approx(0.45)
    assert result['margin'] == pytest.approx(-0.3)


def test_use_batch_makes_the_other_queries_positives_count_as_negatives():
    """``INFONCE_USE_BATCH=True`` (the default) scores each anchor against the whole batch, so the
    second query's positive (0.6) appears among the first query's negatives."""
    result = _infonce()._calculate_metrics(_TWO_QUERIES, _TWO_QUERY_LABELS)
    assert result['mean_pos'] == pytest.approx(0.7)
    assert result['mean_neg'] == pytest.approx(1.4 / 6)
    assert result['margin'] == pytest.approx(0.0)


def test_use_batch_false_keeps_each_query_to_its_own_negatives(monkeypatch):
    """With the batch switched off the same data scores 0.7, because only the query's own (all-zero)
    negatives are considered."""
    monkeypatch.setenv('INFONCE_USE_BATCH', 'False')
    result = _infonce()._calculate_metrics(_TWO_QUERIES, _TWO_QUERY_LABELS)
    assert result['mean_pos'] == pytest.approx(0.7)
    assert result['mean_neg'] == pytest.approx(0.0)
    assert result['margin'] == pytest.approx(0.7)


def test_hard_negatives_truncates_the_negatives(monkeypatch):
    """``INFONCE_HARD_NEGATIVES`` caps how many negatives a query is scored against; the surviving
    negatives here are all orthogonal to the anchor."""
    monkeypatch.setenv('INFONCE_HARD_NEGATIVES', '1')
    result = _infonce()._calculate_metrics(_ONE_QUERY, _ONE_QUERY_LABELS)
    assert result['mean_pos'] == pytest.approx(0.8)
    assert result['mean_neg'] == pytest.approx(0.0)
    assert result['margin'] == pytest.approx(0.8)


def test_the_label_vector_is_one_entry_shorter_per_query_than_the_rows():
    """The label vector marks where each query starts, and the parser shifts every later start by one,
    so k queries need ``rows - k`` labels. A same-length label vector is not rejected -- it silently
    mis-slices, which is why the convention is asserted rather than left to the reader of the parser.

    For the two-query case that means four labels for six rows, and the split lands on 3 + 3.
    """
    result = _infonce()._calculate_metrics(_TWO_QUERIES, _TWO_QUERY_LABELS)
    assert result['mean_pos'] == pytest.approx(0.7)  # both anchors matched their own positive

    same_length = _infonce()._calculate_metrics(_TWO_QUERIES, np.array([1, 0, 0, 1, 0, 0]))
    assert same_length['mean_pos'] == pytest.approx(0.8)  # a different, wrong split -- no error


def test_use_batch_false_with_uneven_queries_scores_them_separately(monkeypatch):
    """The other shape the parser produces: queries of different sizes cannot be stacked into one
    batch, so each is scored against its own negatives only -- here a 0.8 and a 0.6 margin averaging
    to 0.7."""
    monkeypatch.setenv('INFONCE_USE_BATCH', 'False')
    # Query 1 is anchor+positive+2 negatives (4 rows), query 2 is anchor+positive+1 negative (3 rows):
    # 7 rows, 7 - 2 = 5 labels, with the second query starting at index 3.
    rows = np.array([[1., 0], [0.8, 0.6], [0, 1], [0, 0], [1., 0], [0.6, 0.8], [0, 1]])
    result = _infonce()._calculate_metrics(rows, np.array([1, 0, 0, 1, 0]))
    assert result['mean_pos'] == pytest.approx(0.7)
    assert result['mean_neg'] == pytest.approx(0.0)
    assert result['margin'] == pytest.approx(0.7)


def test_compute_metrics_reads_an_eval_prediction_directly():
    """The HF hook path: the whole validation set arrives at once, without ``update``."""
    prediction = EvalPrediction(
        predictions=np.array([[1., 0], [0.8, 0.6], [0, 1], [0, 0]]),
        label_ids=np.array([1, 0, 0, 0]))
    result = _infonce().compute_metrics(prediction)
    assert result['margin'] == pytest.approx(0.8)
    assert result['mean_neg'] == pytest.approx(0.0)
    assert result['mean_pos'] == pytest.approx(0.8)


def test_the_mixin_concatenates_batches_before_computing():
    """``update`` is called once per batch and ``compute`` must see them as one evaluation. The mixin
    has no scoring of its own, so ``InfonceMetrics`` stands in as the concrete subclass."""
    import torch

    metric = InfonceMetrics(_Args(), object())
    metric.update(torch.tensor([[1., 0], [0.8, 0.6]]), torch.tensor([1, 0]))
    metric.update(torch.tensor([[0., 1], [0., 0]]), torch.tensor([0, 0]))
    # Concatenated: one query of anchor + positive + two negatives, same as _ONE_QUERY.
    result = metric.compute()
    assert result['margin'] == pytest.approx(0.8)
    assert result['mean_neg'] == pytest.approx(0.0)
    assert result['mean_pos'] == pytest.approx(0.8)


def test_the_embedding_mixin_accumulates_batches_and_resets():
    """The mixin is what turns per-batch hidden states into one evaluation, so it must survive more
    than one ``update`` and clear on ``reset``."""
    import torch

    metric = EmbedddingMetricMixin()
    metric.update(torch.zeros(2, 4), torch.tensor([1, 0]))
    metric.update(torch.ones(2, 4), torch.tensor([1, 0]))
    assert len(metric.last_hidden_state) == 2
    assert len(metric.labels) == 2

    metric.reset()
    assert metric.last_hidden_state == []
    assert metric.labels == []


# ---------------------------------------------------------------------------
# PairedMetrics (needs scikit-learn)
# ---------------------------------------------------------------------------

def test_paired_metrics_correlates_the_labels_with_each_distance():
    """When the labels *are* the cosine scores, the cosine correlation must be exactly 1: that is what
    makes the other distances comparable to it."""
    pytest.importorskip('sklearn', reason='PairedMetrics uses sklearn paired distance functions')
    from swift.metrics.embedding import PairedMetrics

    # Interleaved rows: (anchor, pair) per query, as _parse_pair_sentence splits even/odd rows.
    predictions = np.array([[1., 0], [1, 0], [1, 0], [0, 1], [1, 0], [0.6, 0.8]])
    labels = np.array([1., 0, 0.6])  # the cosine similarities of the three pairs
    result = PairedMetrics(_Args(), object())._calculate_metrics(predictions, labels)

    assert set(result) == {
        'pearson_cosine', 'pearson_euclidean', 'pearson_manhattan', 'pearson_dot_product',
        'spearman_cosine', 'spearman_euclidean', 'spearman_manhattan', 'spearman_dot_product',
    }
    assert result['pearson_cosine'] == pytest.approx(1.0)
    assert result['spearman_cosine'] == pytest.approx(1.0)


def test_paired_metrics_reads_an_eval_prediction_directly():
    """The HF hook path, for the paired metric as well."""
    pytest.importorskip('sklearn', reason='PairedMetrics uses sklearn paired distance functions')
    from swift.metrics.embedding import PairedMetrics

    prediction = EvalPrediction(
        predictions=np.array([[1., 0], [1, 0], [1, 0], [0, 1], [1, 0], [0.6, 0.8]]),
        label_ids=np.array([1., 0, 0.6]))
    result = PairedMetrics(_Args(), object()).compute_metrics(prediction)
    assert result['pearson_cosine'] == pytest.approx(1.0)

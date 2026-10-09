# Copyright (c) ModelScope Contributors. All rights reserved.
"""``compute_acc`` and ``AccMetrics``: the token/sequence accuracy behind ``--eval_metric acc``.

Two things are easy to get wrong here and are therefore pinned rather than described. The first is the
shift: for a causal LM the logit at position *i* predicts the label at position *i+1*, so predictions
and labels are offset before they are compared -- an encoder-decoder model must NOT be offset. The
second is what counts in the denominator: ``-100`` labels are padding and are dropped, so the accuracy
of a padded batch is not diluted by its padding.

``AccMetrics`` is an ``EvalMetrics``, not a ``Metric``: it has no accumulate/reset state because the
HF ``Trainer`` hands it the whole validation set at once through ``compute_metrics``.
"""
import numpy as np
import pytest
import torch
from transformers.trainer_utils import EvalPrediction

from swift.metrics.acc import AccMetrics, compute_acc


class _Args:
    acc_strategy = 'token'


class _Trainer:
    is_encoder_decoder = False


def _acc_metrics(acc_strategy='token', is_encoder_decoder=False):
    args = _Args()
    args.acc_strategy = acc_strategy
    trainer = _Trainer()
    trainer.is_encoder_decoder = is_encoder_decoder
    return AccMetrics(args, trainer)


def _eval_prediction(predictions, label_ids):
    return EvalPrediction(predictions=np.asarray(predictions), label_ids=np.asarray(label_ids))


# ---------------------------------------------------------------------------
# The shift
# ---------------------------------------------------------------------------

def test_a_causal_lm_is_scored_against_the_next_token():
    """Prediction at position i is compared with the label at position i+1, so a batch whose logits
    are the labels shifted by one is perfectly accurate."""
    metric = _acc_metrics().compute_metrics(_eval_prediction([[1, 2, 3, 0]], [[0, 1, 2, 3]]))
    assert metric == {'token_acc': pytest.approx(1.0)}


def test_torch_tensors_are_accepted_as_inputs():
    """``SwiftMixin._compute_acc`` calls this from the training loop with tensors (the argmax output),
    not numpy -- and a floating-point *tensor* label must be refused before the numpy checks run."""
    assert compute_acc(torch.tensor([[1, 2, 3, 0]]), torch.tensor([[0, 1, 2, 3]])) == {'token_acc': [True, True, True]}
    assert compute_acc(torch.tensor([[1, 2]]), torch.tensor([[1., 2.]])) == {}


def test_an_encoder_decoder_model_is_not_shifted():
    """A seq2seq model predicts its own next token in place; offsetting it would misalign every label."""
    metric = _acc_metrics(is_encoder_decoder=True).compute_metrics(_eval_prediction([[1, 2, 3]], [[1, 2, 3]]))
    assert metric == {'token_acc': pytest.approx(1.0)}


def test_accuracy_of_a_completely_wrong_batch_is_zero():
    metric = _acc_metrics().compute_metrics(_eval_prediction([[9, 9, 9, 9]], [[0, 1, 2, 3]]))
    assert metric == {'token_acc': pytest.approx(0.0)}


def test_accuracy_of_a_half_right_batch_is_a_half():
    metric = _acc_metrics().compute_metrics(_eval_prediction([[5, 9, 7]], [[0, 5, 6]]))
    assert metric == {'token_acc': pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# What the denominator is
# ---------------------------------------------------------------------------

def test_minus_100_labels_are_dropped_from_the_denominator():
    """Two of three positions are valid, one of them right: 0.5, not 0.33."""
    assert compute_acc(np.array([1, 2, 3]), np.array([1, -100, 9])) == {'acc': [True, False]}


def test_a_batch_that_is_all_padding_yields_no_observations():
    """An empty denominator must produce an empty observation list, not a division by zero or a
    fabricated 0%."""
    assert compute_acc(np.array([1, 2]), np.array([-100, -100])) == {'acc': []}


@pytest.mark.xfail(
    reason='AccMetrics.compute_metrics averages an empty list: its `len(metric) == 0` guard counts dict '
    'keys rather than observations, so an all-padding validation batch raises ZeroDivisionError',
    strict=False)
def test_a_validation_set_that_is_all_padding_does_not_divide_by_zero():
    metrics = _acc_metrics().compute_metrics(_eval_prediction([[1, 2]], [[-100, -100]]))
    assert metrics == {}


@pytest.mark.parametrize('predictions,label_ids', [
    ([[1, 2]], [[1., 2.]]),  # float labels are a classification head, not a generation
    ([[1, 2, 3]], [[1, 2]]),  # shapes that cannot be lined up
])
def test_inputs_that_cannot_be_scored_are_skipped(predictions, label_ids):
    assert compute_acc(np.asarray(predictions), np.asarray(label_ids)) == {}
    assert _acc_metrics().compute_metrics(_eval_prediction(predictions, label_ids)) == {}


def test_a_1d_prediction_is_single_label_classification():
    """1-D predictions have no position axis, so they are compared elementwise and reported as ``acc``."""
    assert compute_acc(np.array([1, 2, 3]), np.array([1, 0, 3])) == {'acc': [True, False, True]}


# ---------------------------------------------------------------------------
# Sequence strategy
# ---------------------------------------------------------------------------

def test_seq_strategy_scores_a_whole_sequence_as_one_observation():
    """``--acc_strategy seq`` counts a sequence only if every token in it is right, and reports the
    mean over sequences rather than over tokens."""
    metric = _acc_metrics(acc_strategy='seq').compute_metrics(
        _eval_prediction([[1, 2, 3, 0], [0, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]]))
    assert metric == {'seq_acc': pytest.approx(0.5)}


def test_seq_strategy_with_cu_seqlens_scores_each_packed_sequence_separately():
    """Under padding-free packing the row is several sequences, so ``cu_seqlens`` is what delimits them."""
    result = compute_acc(
        np.array([[1, 2, 3, 0]]),
        np.array([[0, 1, 2, 3]]),
        acc_strategy='seq',
        cu_seqlens=np.array([0, 2, 3]))
    assert list(result) == ['seq_acc']
    assert [bool(x) for x in result['seq_acc']] == [True, True]


def test_cu_seqlens_on_a_multi_row_batch_falls_back_to_per_row_scoring():
    """``cu_seqlens`` only describes a single packed row; with more rows the metric scores each row
    as one sequence instead of silently indexing the wrong one."""
    result = compute_acc(
        np.array([[1, 2, 3, 0], [0, 1, 2, 3]]),
        np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        acc_strategy='seq',
        cu_seqlens=np.array([0, 2, 3]))
    assert list(result) == ['seq_acc']
    assert [bool(x) for x in result['seq_acc']] == [True, False]


# ---------------------------------------------------------------------------
# AccMetrics
# ---------------------------------------------------------------------------

def test_preprocess_logits_for_metrics_argmaxes_and_unwraps_a_sequence():
    """The Trainer calls this on every batch, so it must both reduce logits to ids and tolerate a model
    that returns ``(logits, ...)``."""
    metrics = _acc_metrics()
    logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2]]])
    assert metrics.preprocess_logits_for_metrics(logits, None).tolist() == [[1, 0]]
    assert metrics.preprocess_logits_for_metrics((logits, 'aux'), None).tolist() == [[1, 0]]

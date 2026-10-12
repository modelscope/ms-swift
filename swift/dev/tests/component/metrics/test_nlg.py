# Copyright (c) ModelScope Contributors. All rights reserved.
"""``compute_rouge_bleu`` and ``NlgMetrics``: the generation metric behind ``--eval_metric nlg``.

The scores are reported as percentages (``0..100``), and a pair that cannot be scored -- empty text,
or text that tokenizes to nothing -- is skipped rather than counted as a zero. That distinction is the
whole reason these cases are pinned: a skipped pair leaves the denominator alone, whereas counting it
as 0% would drag every reported average down.

``NlgMetrics`` receives its predictions as *serialized* rows (``Serializer`` pickles each string into a
tensor), which is what the Trainer hands over; the round-trip is exercised here so a change to the
serialization cannot silently turn the metric into a no-op.
"""
import numpy as np
import pytest
import torch
from transformers.trainer_utils import EvalPrediction

from swift.metrics.nlg import NlgMetrics, compute_rouge_bleu
from swift.utils import Serializer

pytest.importorskip('jieba', reason='compute_rouge_bleu tokenizes with jieba')
pytest.importorskip('nltk', reason='compute_rouge_bleu scores BLEU with nltk')
pytest.importorskip('rouge', reason='compute_rouge_bleu scores ROUGE with the rouge package')

_SCORES = {'rouge-1', 'rouge-2', 'rouge-l', 'bleu-4'}


class _Args:
    pass


def _pack(texts):
    """Serialize a list of strings the way the Trainer does: one padded row per text."""
    tensors = [Serializer.to_tensor(text) for text in texts]
    width = max(tensor.numel() for tensor in tensors)
    padded = [torch.nn.functional.pad(tensor, (0, width - tensor.numel())) for tensor in tensors]
    return torch.stack(padded).numpy()


def test_identical_text_scores_full_marks():
    result = compute_rouge_bleu(['我喜欢吃苹果'], ['我喜欢吃苹果'])
    assert set(result) == _SCORES
    assert result['rouge-1'] == pytest.approx(100.0)
    assert result['rouge-2'] == pytest.approx(100.0)
    assert result['rouge-l'] == pytest.approx(100.0)
    assert 0 < result['bleu-4'] <= 100


def test_unrelated_text_scores_zero():
    result = compute_rouge_bleu(['苹果'], ['香蕉'])
    assert result['rouge-1'] == pytest.approx(0.0)
    assert result['rouge-l'] == pytest.approx(0.0)
    assert result['bleu-4'] == pytest.approx(0.0)


def test_partial_overlap_lands_between_the_two_extremes():
    result = compute_rouge_bleu(['今天天气很好'], ['今天天气很差'])
    assert 0 < result['rouge-1'] < 100


def test_nothing_to_score_reports_zeros_rather_than_dividing_by_zero():
    assert compute_rouge_bleu([], []) == dict.fromkeys(_SCORES, 0)


def test_a_prediction_that_tokenizes_to_nothing_is_skipped():
    """Blank text has no hypothesis to score, so the pair drops out instead of counting as 0%."""
    assert compute_rouge_bleu(['   '], ['苹果']) == dict.fromkeys(_SCORES, 0)


def test_nlg_metrics_decodes_serialized_rows():
    """The end-to-end path: serialized predictions in, percentage scores out."""
    metrics = NlgMetrics(_Args(), object())
    prediction = EvalPrediction(
        predictions=_pack(['我喜欢吃苹果', '今天天气很好']),
        label_ids=_pack(['我喜欢吃苹果', '今天天气很差']))
    result = metrics.compute_metrics(prediction)

    assert set(result) == _SCORES
    assert all(0 <= value <= 100 for value in result.values())
    # One of the two rows is an exact match, so the pair is not scoring as if both were wrong.
    assert result['rouge-1'] > compute_rouge_bleu(['今天天气很好'], ['今天天气很差'])['rouge-1']


def test_serializer_round_trips_a_row():
    """``NlgMetrics`` is only as good as this round-trip; an empty string is the edge case."""
    for text in ('', '苹果', '我喜欢吃苹果'):
        assert Serializer.from_tensor(Serializer.to_tensor(text)) == text


def test_serialized_rows_of_uneven_length_are_padded_not_truncated():
    """Real batches hold different-length strings, so the reader must stop at the recorded length."""
    packed = _pack(['苹果', '我喜欢吃苹果'])
    assert Serializer.from_tensor(packed[0]) == '苹果'
    assert Serializer.from_tensor(packed[1]) == '我喜欢吃苹果'
    assert packed.dtype == np.uint8

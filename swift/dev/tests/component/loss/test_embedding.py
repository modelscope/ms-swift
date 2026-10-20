# Copyright (c) ModelScope Contributors. All rights reserved.
"""The embedding losses behind ``--loss_type cosine_similarity|contrastive|online_contrastive|infonce``.

``_parse_pair_sentence`` defines the data contract for the pair losses -- rows are interleaved
anchor/pair, and a 3-D hidden state is CLS-pooled -- and ``_parse_multi_negative_sentences`` defines it
for infonce: a query is an anchor followed by its positive and its negatives, with the boundaries
recovered from the label vector. ``InfonceLoss`` is then driven through its configuration matrix
(in-batch switch, temperature, hard negatives, the Qwen3-style q-q/d-d components, fake-negative
masking), each against a probed value, because what a switch *means* is exactly what the number
encodes: masking a fake negative lowers the loss, a negative outscoring the positive raises it.

The distributed in-batch path is recorded as xfail: it crashes for world_size > 1 on the plain
(non-Megatron, non-sequence-parallel) route.
"""
import math

import numpy as np
import pytest
import torch

from swift.loss.embedding import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    InfonceLoss,
    OnlineContrastiveLoss,
    SiameseDistanceMetric,
    _parse_multi_negative_sentences,
    _parse_pair_sentence,
)


class _Trainer:
    pass


def _loss(cls):
    return cls(args=None, trainer=_Trainer())


def _unit(c):
    """A unit 2-vector whose cosine against [1, 0] is c, so their cosine distance is 1 - c."""
    return [c, math.sqrt(1 - c * c)]


# ---------------------------------------------------------------------------
# The pair contract: _parse_pair_sentence and the distance metrics
# ---------------------------------------------------------------------------

def test_pair_rows_are_split_even_odd():
    hidden = torch.tensor([[1., 0], [0, 1], [1, 1], [0, 0]])
    first, second = _parse_pair_sentence(hidden)
    assert first.tolist() == [[1., 0], [1., 1]]
    assert second.tolist() == [[0., 1], [0., 0]]


def test_pair_parsing_accepts_a_model_output_dict():
    hidden = torch.tensor([[1., 0], [0, 1], [1, 1], [0., 0]])
    first, second = _parse_pair_sentence({'last_hidden_state': hidden})
    assert first.tolist() == [[1., 0], [1., 1]]
    assert second.tolist() == [[0., 1], [0., 0]]


def test_a_3d_hidden_state_is_cls_pooled():
    hidden = torch.arange(12, dtype=torch.float).reshape(2, 3, 2).repeat(2, 1, 1)
    first, second = _parse_pair_sentence(hidden)
    assert first.tolist() == [[0., 1], [0., 1]]  # CLS (position 0) of rows 0 and 2
    assert second.tolist() == [[6., 7], [6., 7]]  # CLS of rows 1 and 3


def test_the_siamese_distance_metrics():
    x, y = torch.tensor([[1., 0]]), torch.tensor([[0., 1]])
    assert float(SiameseDistanceMetric.EUCLIDEAN(x, y)) == pytest.approx(math.sqrt(2), abs=1e-6)
    assert float(SiameseDistanceMetric.MANHATTAN(x, y)) == pytest.approx(2.0)
    assert float(SiameseDistanceMetric.COSINE_DISTANCE(x, y)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CosineSimilarityLoss / ContrastiveLoss / OnlineContrastiveLoss
# ---------------------------------------------------------------------------

# Two pairs: identical (cosine 1) and orthogonal (cosine 0).
_PAIRS = torch.tensor([[1., 0], [1, 0], [1, 0], [0, 1]])


def test_cosine_similarity_loss_is_zero_when_the_labels_are_the_cosines():
    loss = _loss(CosineSimilarityLoss)(_PAIRS, torch.tensor([1., 0]))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_loss_is_the_mse_between_cosines_and_labels():
    loss = _loss(CosineSimilarityLoss)(_PAIRS, torch.tensor([1., 1]))
    assert float(loss) == pytest.approx(0.5, abs=1e-6)  # ((1-1)^2 + (0-1)^2) / 2


def test_contrastive_loss_is_zero_when_positives_touch_and_negatives_are_apart():
    """An identical positive pair (distance 0) and an orthogonal negative pair (distance 1) are both
    free: the margin term only charges negatives closer than 0.5."""
    loss = _loss(ContrastiveLoss)(_PAIRS, torch.tensor([1., 0]))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_contrastive_loss_charges_a_distant_positive_and_a_close_negative():
    hidden = torch.tensor([[1., 0], [0, 1], [1, 0], [1, 0]])  # positive orthogonal, negative identical
    loss = _loss(ContrastiveLoss)(hidden, torch.tensor([1., 0]))
    # 0.5 * (1^2 + relu(0.5 - 0)^2) averaged over the two pairs = (0.5 + 0.125) / 2
    assert float(loss) == pytest.approx(0.3125, abs=1e-6)


def test_online_contrastive_loss_mines_the_hard_pairs():
    """Positive distances [0.8, 0.2], negative distances [0.1, 0.9]: every pair survives mining on its
    own side, giving 0.8^2 + 0.2^2 + relu(0.5 - 0.1)^2 = 0.84."""
    hidden = torch.tensor([
        [1., 0], _unit(0.2),  # positive pair, cosine distance 0.8
        [1., 0], _unit(0.8),  # positive pair, cosine distance 0.2
        [1., 0], _unit(0.9),  # negative pair, cosine distance 0.1
        [1., 0], _unit(0.1),  # negative pair, cosine distance 0.9
    ])
    loss = _loss(OnlineContrastiveLoss)(hidden, torch.tensor([1., 1, 0, 0]))
    assert float(loss) == pytest.approx(0.84, abs=1e-4)


def test_online_contrastive_loss_selects_nothing_from_one_pair_of_each():
    """With a single positive and a single negative the mining thresholds are the pair's own mean, and
    a distance is never strictly beyond itself: nothing is selected, so the loss is 0."""
    hidden = torch.tensor([[1., 0], [0, 1], [1., 0], [1, 0]])
    loss = _loss(OnlineContrastiveLoss)(hidden, torch.tensor([1., 0]))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The infonce contract: _parse_multi_negative_sentences
# ---------------------------------------------------------------------------

def test_a_single_query_spans_to_the_end_of_the_batch():
    sentences = torch.arange(5.).reshape(5, 1).expand(5, 2)
    parts = _parse_multi_negative_sentences(sentences, torch.tensor([1, 0, 0, 0]), None)
    assert [part.shape[0] for part in parts] == [5]


def test_hard_negatives_truncates_a_query_with_more_negatives():
    sentences = torch.arange(5.).reshape(5, 1).expand(5, 2)
    parts = _parse_multi_negative_sentences(sentences, torch.tensor([1, 0, 0, 0]), 1)
    assert [part.shape[0] for part in parts] == [3]  # anchor + positive + one negative
    assert parts[0][0, 0] == 0  # the group still starts at the anchor


def test_hard_negatives_pads_a_query_with_fewer_negatives():
    sentences = torch.arange(3.).reshape(3, 1).expand(3, 2)  # anchor + positive + one negative
    np.random.seed(0)  # the padding duplicates a negative at random
    parts = _parse_multi_negative_sentences(sentences, torch.tensor([1, 0, 0]), 2)
    assert [part.shape[0] for part in parts] == [4]


def test_hard_negatives_matching_exactly_leaves_the_query_untouched():
    """A group of anchor + positive + two negatives against hard_negatives=2: neither the truncate
    nor the pad branch runs."""
    sentences = torch.arange(5.).reshape(5, 1).expand(5, 2)
    parts = _parse_multi_negative_sentences(sentences, torch.tensor([1, 0, 0]), 2)
    assert [part.shape[0] for part in parts] == [4]  # anchor + positive + both negatives


def test_a_query_without_negatives_is_refused():
    sentences = torch.arange(2.).reshape(2, 1).expand(2, 2)
    with pytest.raises(AssertionError):
        _parse_multi_negative_sentences(sentences, torch.tensor([1, 0]), 1)


# ---------------------------------------------------------------------------
# InfonceLoss: one query
# ---------------------------------------------------------------------------

# A query: an anchor, its positive (cosine 0.8), and two orthogonal negatives.
_ONE_QUERY = {
    'last_hidden_state': torch.tensor([[1., 0, 0, 0], [0.8, 0.6, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])}
_ONE_QUERY_LABELS = torch.tensor([1, 0, 0, 0])
# The same query with a negative that outscores the positive (0.9 vs 0.6).
_ONE_QUERY_HARD = {
    'last_hidden_state': torch.tensor([[1., 0, 0, 0], [0.6, 0.8, 0, 0], [0.9, 0.435889894, 0, 0], [0, 0, 1, 0]])}


def test_infonce_loss_on_a_clean_query_is_near_zero():
    loss = _loss(InfonceLoss)(_ONE_QUERY, _ONE_QUERY_LABELS)
    assert 0 < float(loss) < 0.01


def test_infonce_loss_rises_when_a_negative_outscores_the_positive():
    clean = float(_loss(InfonceLoss)(_ONE_QUERY, _ONE_QUERY_LABELS))
    hard = float(_loss(InfonceLoss)(_ONE_QUERY_HARD, _ONE_QUERY_LABELS))
    assert hard == pytest.approx(3.0487, rel=1e-3)
    assert hard > clean


def test_a_sharper_temperature_sharpens_a_correct_ranking_to_zero(monkeypatch):
    monkeypatch.setenv('INFONCE_TEMPERATURE', '0.01')
    loss = _loss(InfonceLoss)(_ONE_QUERY, _ONE_QUERY_LABELS)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_use_batch_false_scores_a_lone_query_identically(monkeypatch):
    """With a single query there are no other queries to treat as negatives, so the switch is a
    no-op."""
    default = float(_loss(InfonceLoss)(_ONE_QUERY, _ONE_QUERY_LABELS))
    monkeypatch.setenv('INFONCE_USE_BATCH', 'False')
    assert float(_loss(InfonceLoss)(_ONE_QUERY, _ONE_QUERY_LABELS)) == pytest.approx(default, rel=1e-6)


def test_hard_negatives_truncates_the_scored_candidates(monkeypatch):
    monkeypatch.setenv('INFONCE_HARD_NEGATIVES', '1')
    loss = _loss(InfonceLoss)(_ONE_QUERY, _ONE_QUERY_LABELS)
    # only the anchor, the positive and one (zero-scoring) negative remain
    assert float(loss) == pytest.approx(0.00033540, rel=1e-3)


def test_the_qq_block_is_a_no_op_for_a_lone_query(monkeypatch):
    """The q-q block excludes the query itself, so with one query the block is entirely -inf."""
    monkeypatch.setenv('INFONCE_INCLUDE_QQ', 'True')
    loss = _loss(InfonceLoss)(_ONE_QUERY, _ONE_QUERY_LABELS)
    assert float(loss) == pytest.approx(0.00067068, rel=1e-3)


def test_the_dd_block_adds_document_document_candidates(monkeypatch):
    monkeypatch.setenv('INFONCE_INCLUDE_DD', 'True')
    loss = _loss(InfonceLoss)(_ONE_QUERY, _ONE_QUERY_LABELS)
    assert float(loss) == pytest.approx(0.12781, rel=1e-3)


# ---------------------------------------------------------------------------
# InfonceLoss: two queries
# ---------------------------------------------------------------------------

# Two queries of anchor + positive + negative. The label vector is one entry shorter per query.
_TWO_QUERIES = {
    'last_hidden_state': torch.tensor([[1., 0], [0.8, 0.6], [0, 1], [1., 0], [0.6, 0.8], [0, 1]])}
_TWO_QUERIES_LABELS = torch.tensor([1, 0, 1, 0])
# Query 1 is anchor + positive + two negatives (4 rows), query 2 is anchor + positive + one negative
# (3 rows): 7 rows, 7 - 2 = 5 labels.
_UNEVEN = {
    'last_hidden_state': torch.tensor([[1., 0], [0.8, 0.6], [0, 1], [0, 0], [1., 0], [0.6, 0.8], [0, 1]])}
_UNEVEN_LABELS = torch.tensor([1, 0, 0, 1, 0])


def test_in_batch_negatives_include_the_other_queries_positive():
    loss = _loss(InfonceLoss)(_TWO_QUERIES, _TWO_QUERIES_LABELS)
    assert float(loss) == pytest.approx(1.1275, rel=1e-3)


def test_use_batch_false_keeps_the_queries_independent(monkeypatch):
    """With the batch off, a query is never scored against the other query's positive, so the loss
    collapses to the per-query ranking error."""
    monkeypatch.setenv('INFONCE_USE_BATCH', 'False')
    loss = _loss(InfonceLoss)(_TWO_QUERIES, _TWO_QUERIES_LABELS)
    assert float(loss) == pytest.approx(0.0014055, rel=1e-3)


def test_the_qq_block_adds_the_other_anchors_as_candidates(monkeypatch):
    monkeypatch.setenv('INFONCE_INCLUDE_QQ', 'True')
    loss = _loss(InfonceLoss)(_TWO_QUERIES, _TWO_QUERIES_LABELS)
    assert float(loss) == pytest.approx(3.1430, rel=1e-3)


def test_the_dd_block_adds_the_other_positives_as_candidates(monkeypatch):
    monkeypatch.setenv('INFONCE_INCLUDE_DD', 'True')
    loss = _loss(InfonceLoss)(_TWO_QUERIES, _TWO_QUERIES_LABELS)
    assert float(loss) == pytest.approx(2.9702, rel=1e-3)


def test_masking_fake_negatives_drops_the_outscores_positive_candidate(monkeypatch):
    """Query 2's own positive scores 0.6, but query 1's positive scores 0.8 against the same anchor --
    a 'fake negative' above the 0.6 + 0.1 margin. Masking it collapses the loss from 1.13 to 0.066."""
    monkeypatch.setenv('INFONCE_MASK_FAKE_NEGATIVE', 'True')
    loss = _loss(InfonceLoss)(_TWO_QUERIES, _TWO_QUERIES_LABELS)
    assert float(loss) == pytest.approx(0.066232, rel=1e-3)


def test_all_qwen3_components_together(monkeypatch):
    monkeypatch.setenv('INFONCE_INCLUDE_QQ', 'True')
    monkeypatch.setenv('INFONCE_INCLUDE_DD', 'True')
    monkeypatch.setenv('INFONCE_MASK_FAKE_NEGATIVE', 'True')
    loss = _loss(InfonceLoss)(_TWO_QUERIES, _TWO_QUERIES_LABELS)
    assert float(loss) == pytest.approx(2.9043, rel=1e-3)


def test_uneven_queries_without_the_batch_are_scored_in_a_loop(monkeypatch):
    monkeypatch.setenv('INFONCE_USE_BATCH', 'False')
    loss = _loss(InfonceLoss)(_UNEVEN, _UNEVEN_LABELS)
    assert float(loss) == pytest.approx(0.0015732, rel=1e-3)


def test_uneven_queries_with_the_batch_score_against_every_document():
    loss = _loss(InfonceLoss)(_UNEVEN, _UNEVEN_LABELS)
    assert float(loss) == pytest.approx(1.1278, rel=1e-3)


def test_uneven_queries_with_every_component_on(monkeypatch):
    """The per-query loop with the q-q, d-d and masking branches, which only this shape reaches."""
    monkeypatch.setenv('INFONCE_INCLUDE_QQ', 'True')
    monkeypatch.setenv('INFONCE_INCLUDE_DD', 'True')
    monkeypatch.setenv('INFONCE_MASK_FAKE_NEGATIVE', 'True')
    loss = _loss(InfonceLoss)(_UNEVEN, _UNEVEN_LABELS)
    assert float(loss) == pytest.approx(2.9044, rel=1e-3)


# ---------------------------------------------------------------------------
# InfonceLoss: the distributed in-batch path
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason='the plain (non-Megatron, non-sequence-parallel) in-batch path stores the local 2-D '
    '`sentences` into a list of 3-D per-rank tensors, so torch.cat raises for world_size > 1',
    strict=False)
def test_in_batch_negatives_gather_across_ranks(monkeypatch):
    """In-batch negatives are supposed to come from every rank: the local loss must equal the loss on
    the gathered batch."""
    import swift.loss.embedding as embedding

    local = {'last_hidden_state': torch.tensor([[1., 0], [0.8, 0.6], [0, 1]])}
    other = torch.tensor([[1., 0], [0.6, 0.8], [0, 1]])
    local_labels = torch.tensor([1, 0, 0])
    other_labels = torch.tensor([1, 0, 0])

    monkeypatch.setattr(embedding, 'get_dist_setting', lambda: (0, 0, 2, 2))
    monkeypatch.setattr(
        embedding, 'gather_object',
        lambda obj: ([local['last_hidden_state'].unsqueeze(0), other.unsqueeze(0)]
                     if obj.dim() == 3 else [local_labels, other_labels]))

    gathered = _loss(InfonceLoss)(
        {'last_hidden_state': torch.cat([local['last_hidden_state'], other])},
        torch.cat([local_labels, other_labels]))
    assert float(_loss(InfonceLoss)(local, local_labels)) == pytest.approx(float(gathered), rel=1e-4)

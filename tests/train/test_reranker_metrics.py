from types import SimpleNamespace

from swift.metrics.reranker import RerankerMetrics


def _build_metrics(loss_type):
    metrics = RerankerMetrics.__new__(RerankerMetrics)
    metrics.args = SimpleNamespace(loss_type=loss_type)
    metrics.trainer = None
    return metrics


def test_pointwise_reranker_metrics_support_negative_only_queries():
    metrics = _build_metrics('pointwise_reranker')

    result = metrics._calculate_metrics(
        logits=[-2.0, -1.0, 3.0, -0.5],
        labels=[0, 0, 1, 0],
        group_sizes=[2, 2],
    )

    assert result['acc'] == 1.0
    assert result['precision'] == 1.0
    assert result['recall'] == 1.0
    assert result['f1'] == 1.0
    assert result['query_count'] == 2.0
    assert result['ranking_query_count'] == 1.0
    assert result['negative_only_query_count'] == 1.0
    assert result['mrr'] == 1.0
    assert result['ndcg'] == 1.0


def test_listwise_reranker_metrics_preserve_group_boundaries():
    metrics = _build_metrics('listwise_reranker')

    result = metrics._calculate_metrics(
        logits=[1.0, 0.0, -1.0, 1.2, 1.0, 0.5],
        labels=[1, 0, 0, 0, 1, 0],
        group_sizes=[3, 3],
    )

    assert result['query_count'] == 2.0
    assert result['ranking_query_count'] == 2.0
    assert result['negative_only_query_count'] == 0.0
    assert result['mrr'] == 0.75

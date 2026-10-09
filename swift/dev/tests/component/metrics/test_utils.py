# Copyright (c) ModelScope Contributors. All rights reserved.
"""``Metric``, ``MeanMetric`` and ``InferStats``: the primitives the rest of ``swift.metrics`` builds on.

``Metric`` is the three-phase contract the package shares -- ``update`` accumulates, ``compute`` reads
a snapshot, ``reset`` returns to the declared defaults -- and ``add_state`` is the only reason ``reset``
works without every subclass writing its own. ``MeanMetric`` is the running average that
``SwiftMixin.compute_custom_metrics`` drains at every log step, so its count/mean bookkeeping for each
shape a caller can hand it (scalar, 0-d tensor, 1-D tensor, list, array, nothing at all) is asserted
here rather than assumed. ``eval_metrics_map`` is the ``--eval_metric`` registry, so its names are CLI
surface.
"""
import numpy as np
import pytest
import torch
import torch.distributed as dist

from swift.metrics import InferStats, MeanMetric, Metric, eval_metrics_map
from swift.metrics.base import EvalMetrics


class _Recorder(Metric):
    """A minimal concrete metric: enough to exercise the base class without a real one in the way."""

    def __init__(self, **state):
        super().__init__()
        self.add_state('items', **state)

    def update(self, value):
        self.items.append(value)

    def compute(self):
        return {'items': list(self.items)}


class _Completion:

    def __init__(self, id_, prompt_tokens, completion_tokens):
        self.id = id_
        self.usage = type('Usage', (), {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
        })()


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def test_metric_is_abstract():
    """``update``/``compute`` are the contract; an implementation that skips them is not a metric."""
    with pytest.raises(TypeError, match='abstract'):
        Metric()


def test_add_state_before_super_init_is_refused():
    """Declaring state before ``super().__init__()`` would land nowhere and reset to nothing."""

    class Bad(Metric):

        def __init__(self):
            self.add_state('items', default_factory=list)

        def update(self, value):
            self.items.append(value)

        def compute(self):
            return {}

    with pytest.raises(AttributeError, match=r'super\(\)\.__init__'):
        Bad()


def test_add_state_refuses_mixing_the_two_styles_for_one_name():
    """``default`` and ``default_factory`` write to different dicts, so a name may only use one.

    Re-declaring with the *same* style is allowed (it just overwrites); the guard is against the two
    styles colliding, which would make ``reset`` pick one of them arbitrarily.
    """

    class Mixed(Metric):

        def __init__(self):
            super().__init__()
            self.add_state('items', default=[])
            self.add_state('items', default_factory=list)

        def update(self, value):
            pass

        def compute(self):
            return {}

    with pytest.raises(AssertionError, match='self._default'):
        Mixed()


def test_reset_restores_the_declared_defaults():
    metric = _Recorder(default_factory=list)
    metric.update(1.0)
    metric.update(2.0)
    assert metric.compute() == {'items': [1.0, 2.0]}

    metric.reset()
    assert metric.compute() == {'items': []}


def test_re_declaring_a_name_with_the_same_style_overwrites():
    """The guard only refuses a style collision; a plain re-declaration is the caller changing their
    mind, and the last declaration wins."""
    metric = _Recorder(default_factory=list)
    metric.add_state('items', default_factory=list)  # same style again: allowed
    metric.update('first')
    metric.reset()
    assert metric.compute() == {'items': []}


def test_a_mutable_default_is_shared_across_resets_but_a_factory_is_not():
    """Why ``default_factory`` exists: ``default=[]`` hands back the same list on every reset.

    A metric that declares its list with ``default=`` keeps what it accumulated, so the next log
    step silently reports the previous window's values as well. The factory form is the one that
    actually clears.
    """
    shared = _Recorder(default=[])
    fresh = _Recorder(default_factory=list)
    for metric in (shared, fresh):
        metric.update('first')
        metric.reset()

    assert shared.compute() == {'items': ['first']}
    assert fresh.compute() == {'items': []}


# ---------------------------------------------------------------------------
# MeanMetric
# ---------------------------------------------------------------------------

def test_mean_metric_averages_scalars():
    metric = MeanMetric()
    metric.update(2.0)
    metric.update(4.0)
    assert metric.compute() == {'value': pytest.approx(3.0)}


def test_mean_metric_weights_a_tensor_by_its_length():
    """A 1-D tensor is one *sample of n values*, not one value: 3 per-element values must not outweigh
    a batch of 1."""
    metric = MeanMetric()
    metric.update(torch.tensor([1.0, 3.0]))
    assert metric.compute() == {'value': pytest.approx(2.0)}
    assert metric.count == 2


def test_mean_metric_counts_a_0d_tensor_as_one_sample():
    metric = MeanMetric()
    metric.update(torch.tensor(5.0))
    metric.update(torch.tensor(7.0))
    assert metric.compute() == {'value': pytest.approx(6.0)}
    assert metric.count == 2


def test_mean_metric_accepts_a_list_and_an_array():
    from_list = MeanMetric()
    from_list.update([1.0, 2.0, 3.0])
    assert from_list.compute() == {'value': pytest.approx(2.0)}
    assert from_list.count == 3

    from_array = MeanMetric()
    from_array.update(np.array([1.0, 5.0]))
    assert from_array.compute() == {'value': pytest.approx(3.0)}
    assert from_array.count == 2


def test_an_empty_mean_metric_reports_nan_value_instead_of_dividing_by_zero():
    assert MeanMetric().compute() == {'value': 0}
    assert MeanMetric(nan_value=-1.0).compute() == {'value': -1.0}


def test_mean_metric_reset_clears_the_running_total():
    metric = MeanMetric()
    metric.update(torch.tensor([1.0, 3.0]))
    metric.reset()
    assert metric.state == 0.
    assert metric.count == 0
    assert metric.compute() == {'value': 0}


def test_compute_all_reduces_across_ranks_when_a_process_group_is_active(monkeypatch):
    """With a group initialized, ``compute`` reports the cluster-wide mean, not this rank's share.

    ``all_reduce`` is stubbed to world-size-1 semantics (summing over one rank is the identity), so
    what is pinned is that the reduction is requested with SUM and its outcome is what gets reported;
    the real multi-rank arithmetic belongs to a distributed test.
    """
    calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        calls.append(op)

    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'all_reduce', fake_all_reduce)

    # An explicit device is also part of the constructor surface; the default resolves to the
    # current device, which is a GPU in training.
    metric = MeanMetric(device='cpu')
    metric.update(2.0)
    metric.update(4.0)
    assert metric.compute() == {'value': pytest.approx(3.0)}
    assert calls == [dist.ReduceOp.SUM]


# ---------------------------------------------------------------------------
# InferStats
# ---------------------------------------------------------------------------

def test_infer_stats_counts_samples_and_tokens():
    metric = InferStats()
    metric.update(_Completion('a', 10, 5))
    metric.update(_Completion('b', 20, 7))
    result = metric.compute()
    assert result['num_samples'] == 2
    assert result['num_prompt_tokens'] == 30
    assert result['num_generated_tokens'] == 12
    assert result['runtime'] > 0
    assert result['samples/s'] == pytest.approx(2 / result['runtime'])
    assert result['tokens/s'] == pytest.approx(12 / result['runtime'])


def test_infer_stats_counts_a_repeated_request_id_once():
    """Samples are keyed by request id, so a re-reported id replaces rather than adds to the total."""
    metric = InferStats()
    metric.update(_Completion('a', 10, 5))
    metric.update(_Completion('a', 100, 50))
    result = metric.compute()
    assert result['num_samples'] == 1
    assert result['num_prompt_tokens'] == 100
    assert result['num_generated_tokens'] == 50


def test_infer_stats_reset_restarts_the_window():
    metric = InferStats()
    metric.update(_Completion('a', 10, 5))
    metric.reset()
    result = metric.compute()
    assert result['num_samples'] == 0
    assert result['num_prompt_tokens'] == 0
    assert result['num_generated_tokens'] == 0


# ---------------------------------------------------------------------------
# EvalMetrics and the --eval_metric registry
# ---------------------------------------------------------------------------

def test_eval_metrics_leaves_the_logits_alone_by_default():
    """``preprocess_logits_for_metrics`` is opt-in: a metric that does not need argmax keeps the raw
    logits, which is what ``nlg`` relies on."""

    class Trivial(EvalMetrics):

        def compute_metrics(self, eval_prediction):
            return {}

    logits = torch.zeros(2, 3)
    assert Trivial(args=None, trainer=None).preprocess_logits_for_metrics(logits, None) is logits


def test_eval_metrics_map_is_the_eval_metric_registry():
    """``--eval_metric <name>`` looks the name up here, so the keys and the base class are CLI surface."""
    assert set(eval_metrics_map) == {'acc', 'nlg', 'infonce', 'paired', 'reranker'}
    for name, cls in eval_metrics_map.items():
        assert issubclass(cls, EvalMetrics), name
        assert hasattr(cls, 'compute_metrics'), name

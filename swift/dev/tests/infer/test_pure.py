# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure-function tests for the infer pipeline (no GPU, no model, no dataset loader).

These cover the numeric / planning / mapping helpers whose contracts the rest of the pipeline relies
on: the reranker sigmoid, per-group score normalisation, the easy-prompt filter, batch planning, the
prompt cache key, the legacy-compatible metric, the reward DeviceGroup planner, and the task-type ->
pooling-head map.
"""
import math

import pytest


def test_sigmoid_is_numerically_stable_both_branches():
    from swift.dev.components.infer.component import _sigmoid
    assert abs(_sigmoid(0.0) - 0.5) < 1e-12
    # large positive / negative must not overflow (the two-branch form exists for exactly this)
    assert abs(_sigmoid(50.0) - 1.0) < 1e-9
    assert abs(_sigmoid(-50.0) - 0.0) < 1e-9
    # matches the naive formula in the safe middle range
    for x in (-3.0, -0.5, 0.5, 3.0):
        assert abs(_sigmoid(x) - 1.0 / (1.0 + math.exp(-x))) < 1e-12


def test_normalize_minmax_degenerate_and_nan_immune():
    from swift.dev.components.infer.component import _min_max
    # spread group -> [0, ~1]. The denominator carries a +1e-5 epsilon, so the top lands just under
    # 1.0 (2/2.00001); assert to 1e-4, not exactly.
    out = _min_max([1.0, 2.0, 3.0])
    assert abs(out[0]) < 1e-9 and abs(out[-1] - 1.0) < 1e-4
    # degenerate positive group collapses to min(1.0, value)
    assert _min_max([0.3, 0.3]) == [0.3, 0.3]
    assert _min_max([5.0, 5.0]) == [1.0, 1.0]
    # degenerate non-positive group collapses to 0.0
    assert _min_max([-2.0, -2.0]) == [0.0, 0.0]
    # a single nan must not poison the group's min/max; it passes through untouched
    out = _min_max([1.0, float('nan'), 3.0])
    assert abs(out[0]) < 1e-9 and abs(out[2] - 1.0) < 1e-4
    assert out[1] != out[1]  # still nan
    # empty / all-nan are returned as-is
    assert _min_max([]) == []
    all_nan = _min_max([float('nan'), float('nan')])
    assert all(v != v for v in all_nan)


def test_is_too_easy_threshold_and_disable():
    from swift.dev.components.infer.component import _is_too_easy
    assert _is_too_easy(3, 4, None) is False  # None disables
    assert _is_too_easy(3, 4, 0.75) is True  # 0.75 >= 0.75
    assert _is_too_easy(2, 4, 0.75) is False


def test_plan_batches_keeps_partial_tail_and_respects_max():
    from swift.dev.components.infer.component import _plan_batches
    # 7 rows, batch 3 -> [(0,3),(3,6),(6,7)] : the trailing partial batch is KEPT (unlike legacy)
    assert _plan_batches(7, 3, None) == [(0, 3), (3, 6), (6, 7)]
    # max_batches truncates
    assert _plan_batches(7, 3, 2) == [(0, 3), (3, 6)]
    # batch_size < 1 is fatal, not silently coerced
    with pytest.raises(ValueError, match='batch_size must be >= 1'):
        _plan_batches(7, 0, None)


def test_prompt_key_is_stable_and_order_insensitive_to_dict_keys():
    from swift.dev.components.infer.component import _prompt_key
    a = [{'role': 'user', 'content': 'hi'}]
    b = [{'role': 'user', 'content': 'hi'}]
    assert _prompt_key(a) == _prompt_key(b)
    # a different prompt hashes differently
    assert _prompt_key([{'role': 'user', 'content': 'bye'}]) != _prompt_key(a)
    # dict key order does not matter (sort_keys=True)
    c = [{'content': 'hi', 'role': 'user'}]
    assert _prompt_key(c) == _prompt_key(a)


def test_compute_metric_acc_is_exact_string_match():
    """legacy's ``--metric acc`` is exact string equality, NOT token-level accuracy."""
    from swift.dev.components.infer.component import compute_metric
    results = [
        {'response': '4', 'labels': '4'},
        {'response': 'blue', 'labels': 'red'},
        {'response': 'hello', 'labels': 'hello'},
    ]
    out = compute_metric(results, 'acc')
    # ExactMatch.calculate() -> {'acc': fraction}; 2 of 3 are exact string matches
    assert 'acc' in out and abs(out['acc'] - 2 / 3) < 1e-6, f'unexpected acc payload: {out}'


def test_compute_metric_skips_rows_without_reference():
    from swift.dev.components.infer.component import compute_metric
    # no row has both a response and a labels -> empty dict (and a warning), not a zero score
    assert compute_metric([{'response': 'x'}], 'acc') == {}


def test_reward_device_groups_sizes_each_channel_from_its_own_spec(monkeypatch):
    """The planner's first stage: (name, world_size, gpus_per_worker) per GPU-resident channel.

    Only a scalar RM or an independent generative judge keeps local weights and earns a group; a
    sampler-reusing judge, an API judge and rules need none. The group's width comes from the
    model's OWN parallel_spec (mesh world_size, tp dim as gpus_per_worker) and defaults to the
    sampler's nproc_per_node with one GPU per rank.
    """
    from types import SimpleNamespace

    from swift.dev.components.infer.component import _reward_device_groups
    from swift.dev.rewards.scorer import ChannelPlan, RewardSpec

    def plan(*specs):
        return ChannelPlan(entries=list(specs))

    scalar = RewardSpec('scalar', model_id='rm')
    reuse = RewardSpec('generative_reuse', model_id='base')
    independent = RewardSpec('generative_independent', model_id='judge')
    api = RewardSpec('api', model_id='http://x')

    dist = SimpleNamespace(nproc_per_node=2)
    groups = _reward_device_groups(plan(scalar), plan(independent), dist)
    assert groups == [('orm', 2, 1), ('prm', 2, 1)]  # no spec: sampler width, 1 gpu per rank
    # sampler-reusing judge / API judge keep no local weights -> no group
    assert _reward_device_groups(plan(reuse), plan(api), dist) == []
    assert _reward_device_groups(plan(), plan(), None) == []
    # a channel with its own spec takes the spec's world_size and tp dim
    tp = RewardSpec('scalar', model_id='rm', parallel_spec='tp2dp2')
    assert _reward_device_groups(plan(tp), plan(), None) == [('orm', 4, 2)]
    # no dist config -> width 1
    assert _reward_device_groups(plan(scalar), plan(), None) == [('orm', 1, 1)]


def test_initialize_for_sampling_lays_out_disjoint_rank_blocks(monkeypatch):
    """The planner's second stage: under ray the sampler takes ranks [0, n), each reward channel the
    next disjoint block sized by its own width -- never a uniform ranks * (1 + R)."""
    import sys
    import types

    from swift.dev.components.infer.component import SAMPLER_GROUP, _initialize_for_sampling
    from swift.dev.config import DistributedConfig

    captured = {}

    def fake_initialize(mode=None, nproc_per_node=None, groups=None):
        captured.update(mode=mode, nproc_per_node=nproc_per_node,
                        groups=[(g.name, g.ranks, g.gpus_per_worker) for g in groups])

    fake_twinkle = types.SimpleNamespace(initialize=fake_initialize,
                                         DeviceGroup=None, DeviceGroup_orig=None)
    # _initialize_for_sampling imports DeviceGroup/Platform at call time; patch the real module
    import twinkle
    monkeypatch.setattr(twinkle, 'initialize', fake_initialize, raising=False)

    class _DG:
        def __init__(self, name, ranks, device_type, gpus_per_worker):
            self.name, self.ranks, self.gpus_per_worker = name, ranks, gpus_per_worker
            self.device_type = device_type

    monkeypatch.setattr(twinkle, 'DeviceGroup', _DG, raising=False)

    dist = DistributedConfig(mode='ray', nproc_per_node=2)
    _initialize_for_sampling(dist, [('orm', 2, 1), ('prm', 2, 1)])
    assert captured['mode'] == 'ray'
    assert captured['nproc_per_node'] == 6  # 2 sampler + 2 + 2, not 2*(1+2)
    assert captured['groups'] == [
        (SAMPLER_GROUP, [0, 1], 1),
        ('orm', [2, 3], 1),
        ('prm', [4, 5], 1),
    ]

    # a channel wider than the sampler takes a wider block with its own gpus_per_worker
    _initialize_for_sampling(DistributedConfig(mode='ray', nproc_per_node=1), [('prm', 4, 2)])
    assert captured['groups'] == [(SAMPLER_GROUP, [0], 1), ('prm', [1, 2, 3, 4], 2)]
    assert captured['nproc_per_node'] == 5

    # no reward groups -> just the sampler
    _initialize_for_sampling(DistributedConfig(mode='ray', nproc_per_node=1), [])
    assert captured['groups'] == [(SAMPLER_GROUP, [0], 1)] and captured['nproc_per_node'] == 1

    # ray without nproc_per_node cannot size the sampler group
    with pytest.raises(ValueError, match='nproc_per_node is required'):
        _initialize_for_sampling(DistributedConfig(mode='ray', nproc_per_node=None), [])


def test_pooling_task_map_and_is_pooling():
    from swift.dev.builders import is_pooling_task, pooling_task_for
    assert pooling_task_for('embedding') == 'embed'
    assert pooling_task_for('seq_cls') == 'classify'
    assert pooling_task_for('reranker') == 'classify'
    # generation tasks have no pooling head
    assert pooling_task_for('causal_lm') is None
    assert pooling_task_for('generative_reranker') is None
    assert pooling_task_for(None) is None  # defaults to causal_lm
    assert is_pooling_task('seq_cls') and is_pooling_task('embedding') and is_pooling_task('reranker')
    assert not is_pooling_task('causal_lm') and not is_pooling_task('generative_reranker')
    assert not is_pooling_task(None)

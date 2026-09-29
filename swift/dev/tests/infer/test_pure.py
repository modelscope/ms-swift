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
    from swift.dev.recipe.run_infer import _sigmoid
    assert abs(_sigmoid(0.0) - 0.5) < 1e-12
    # large positive / negative must not overflow (the two-branch form exists for exactly this)
    assert abs(_sigmoid(50.0) - 1.0) < 1e-9
    assert abs(_sigmoid(-50.0) - 0.0) < 1e-9
    # matches the naive formula in the safe middle range
    for x in (-3.0, -0.5, 0.5, 3.0):
        assert abs(_sigmoid(x) - 1.0 / (1.0 + math.exp(-x))) < 1e-12


def test_normalize_minmax_degenerate_and_nan_immune():
    from swift.dev.recipe.run_infer import _normalize
    # spread group -> [0, ~1]. The denominator carries a +1e-5 epsilon, so the top lands just under
    # 1.0 (2/2.00001); assert to 1e-4, not exactly.
    out = _normalize([1.0, 2.0, 3.0])
    assert abs(out[0]) < 1e-9 and abs(out[-1] - 1.0) < 1e-4
    # degenerate positive group collapses to min(1.0, value)
    assert _normalize([0.3, 0.3]) == [0.3, 0.3]
    assert _normalize([5.0, 5.0]) == [1.0, 1.0]
    # degenerate non-positive group collapses to 0.0
    assert _normalize([-2.0, -2.0]) == [0.0, 0.0]
    # a single nan must not poison the group's min/max; it passes through untouched
    out = _normalize([1.0, float('nan'), 3.0])
    assert abs(out[0]) < 1e-9 and abs(out[2] - 1.0) < 1e-4
    assert out[1] != out[1]  # still nan
    # empty / all-nan are returned as-is
    assert _normalize([]) == []
    all_nan = _normalize([float('nan'), float('nan')])
    assert all(v != v for v in all_nan)


def test_is_too_easy_threshold_and_disable():
    from swift.dev.recipe.run_infer import _is_too_easy
    assert _is_too_easy(3, 4, None) is False  # None disables
    assert _is_too_easy(3, 4, 0.75) is True  # 0.75 >= 0.75
    assert _is_too_easy(2, 4, 0.75) is False


def test_plan_batches_keeps_partial_tail_and_respects_max():
    from swift.dev.recipe.run_infer import _plan_batches
    # 7 rows, batch 3 -> [(0,3),(3,6),(6,7)] : the trailing partial batch is KEPT (unlike legacy)
    assert _plan_batches(7, 3, None) == [(0, 3), (3, 6), (6, 7)]
    # max_batches truncates
    assert _plan_batches(7, 3, 2) == [(0, 3), (3, 6)]
    # batch_size < 1 is fatal, not silently coerced
    with pytest.raises(ValueError, match='batch_size must be >= 1'):
        _plan_batches(7, 0, None)


def test_prompt_key_is_stable_and_order_insensitive_to_dict_keys():
    from swift.dev.recipe.run_infer import _prompt_key
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
    from swift.dev.recipe.run_infer import compute_metric
    results = [
        {'response': '4', 'labels': '4'},
        {'response': 'blue', 'labels': 'red'},
        {'response': 'hello', 'labels': 'hello'},
    ]
    out = compute_metric(results, 'acc')
    # ExactMatch.calculate() -> {'acc': fraction}; 2 of 3 are exact string matches
    assert 'acc' in out and abs(out['acc'] - 2 / 3) < 1e-6, f'unexpected acc payload: {out}'


def test_compute_metric_skips_rows_without_reference():
    from swift.dev.recipe.run_infer import compute_metric
    # no row has both a response and a labels -> empty dict (and a warning), not a zero score
    assert compute_metric([{'response': 'x'}], 'acc') == {}


def test_plan_sampling_device_groups_sizes_sampler_plus_each_reward():
    from swift.dev.recipe.run_infer import plan_sampling_device_groups
    # reward_groups is (name, world_size, gpus_per_worker); each channel carries its OWN width, so the
    # total is sampler_ranks + sum(widths), not a uniform ranks * (1 + R).
    groups, total = plan_sampling_device_groups(2, [('orm', 2, 1), ('prm', 2, 1)])
    assert groups == [('sampler', [0, 1], 1), ('orm', [2, 3], 1), ('prm', [4, 5], 1)]
    assert total == 6
    # a channel wider than the sampler takes a wider block, with its own gpus_per_worker
    groups, total = plan_sampling_device_groups(1, [('prm', 4, 2)])
    assert groups == [('sampler', [0], 1), ('prm', [1, 2, 3, 4], 2)] and total == 5
    # no reward groups -> just the sampler
    groups, total = plan_sampling_device_groups(1, [])
    assert groups == [('sampler', [0], 1)] and total == 1
    with pytest.raises(ValueError, match='sampler GPU count must be >= 1'):
        plan_sampling_device_groups(0, [])


def test_exclusive_reward_groups_only_for_gpu_resident_kinds():
    from swift.dev.recipe.run_infer import _RewardModelSpec, _RewardSlot, _exclusive_reward_groups

    def slot(kind, model_id):
        return _RewardSlot(spec=_RewardModelSpec(kind=kind, model_id=model_id))

    scalar = slot('scalar', 'rm')
    reuse = slot('generative_reuse', 'base')
    independent = slot('generative_independent', 'judge')
    api = slot('api', 'http://x')
    rule = _RewardSlot(rule='accuracy')
    # scalar (orm) + independent (prm) each need their own DeviceGroup, in declaration order; with no
    # parallel_spec and no ray config their (width, gpus) fall back to (1, 1).
    assert _exclusive_reward_groups([scalar], [independent], None) == [('orm', 1, 1), ('prm', 1, 1)]
    # a sampler-reusing judge, an API judge and a rule keep no local weights -> no group
    assert _exclusive_reward_groups([reuse, rule], [api], None) == []
    assert _exclusive_reward_groups([], [], None) == []


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


def test_tool_names_extracts_assistant_tool_calls_in_order():
    from swift.dev.recipe.infer_tui import _tool_names
    messages = [
        {'role': 'user', 'content': 'go'},
        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'read_file'}},
                                             {'function': {'name': 'run_command'}}]},
        {'role': 'tool', 'content': 'ok'},
        {'role': 'assistant', 'content': 'done'},  # no tool_calls
    ]
    assert _tool_names(messages) == ['read_file', 'run_command']
    assert _tool_names([{'role': 'assistant', 'content': 'x'}]) == []

# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end pipeline tests over ``run_infer`` with no GPU, no model, no network.

``infer_backend='no'`` builds no sampler and no rollout engine: every candidate is served from
``cache_files``, so the whole generative pipeline -- sample (from cache) -> score -> shape -> write --
runs in-process. ``load_prompt_rows`` is patched to hand back fixed rows, and a callable ORM stands in
for a reward model. This exercises the three orthogonal knobs (num_return_sequences / reward funcs /
output_format) and both writers end to end, which is the surface the slow GPU tests only re-check with a
real engine.
"""
import json
import os

import pytest

from swift.dev.config import DatasetConfig, GenerationConfig, InferConfig, ModelConfig, RLHFConfig, TemplateConfig
from swift.dev.recipe.run_infer import run_infer
from swift.dev.tests.infer.conftest import write_cache_file

P1 = [{'role': 'user', 'content': 'What is 2+2?'}]
P2 = [{'role': 'user', 'content': 'Capital of France?'}]


def _read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _rows():
    return [
        {
            'messages': list(P1),
            'solution': '4'
        },
        {
            'messages': list(P2),
            'solution': 'Paris'
        },
    ]


def _cache(tmp_path):
    """One cache file covering both prompts with three candidates each; the first matches the solution."""
    return write_cache_file(str(tmp_path / 'cache.jsonl'), [(P1, ['4', 'five', '22']), (P2, ['Paris', 'Lyon', 'Mars'])])


def _exact_reward(completions, solution=None, **kwargs):
    """A callable ORM: 1.0 when the completion equals the row's ``solution``, else 0.0.

    ``compute_rewards_per_func`` calls it as ``func(completions, **columns)``, broadcasting the dataset's
    ``solution`` column across the group, so this needs no model and no GPU.
    """
    solution = solution or [''] * len(completions)
    return [1.0 if str(solution[i]).strip() == str(c).strip() else 0.0 for i, c in enumerate(completions)]


# --- 'all' format ---------------------------------------------------------------------


def test_all_format_stores_every_candidate(tmp_path, patch_prompt_rows):
    patch_prompt_rows(_rows())
    out_path = str(tmp_path / 'out.jsonl')
    infer_config = InferConfig(cache_files=[_cache(tmp_path)], num_return_sequences=3, batch_size=2, output_format='all')
    results = run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), backend='no', output_path=out_path)

    assert len(results) == 2
    r0 = results[0]
    assert r0['response'] == '4'  # the first candidate
    assert r0['responses'] == ['4', 'five', '22']  # the whole group
    assert r0['messages'] == P1 + [{'role': 'assistant', 'content': '4'}]
    assert r0['solution'] == '4'  # the dataset column rides along
    assert 'scores' not in r0  # no reward func -> no scores column
    assert _read_jsonl(out_path) == results  # the incremental writer flushed every row


def test_all_format_scores_with_callable_reward(tmp_path, patch_prompt_rows):
    """A reward func scores every candidate even in 'all' format; scoring is orthogonal to storage."""
    patch_prompt_rows(_rows())
    infer_config = InferConfig(cache_files=[_cache(tmp_path)], num_return_sequences=3, batch_size=2, output_format='all')
    results = run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), rlhf_config=RLHFConfig(reward_funcs=[_exact_reward]), backend='no')

    # only the first candidate matches the solution, and normalize_rewards is off by default
    assert results[0]['scores'] == [1.0, 0.0, 0.0]
    assert results[1]['scores'] == [1.0, 0.0, 0.0]


def test_all_format_metric_acc_against_reference(tmp_path, patch_prompt_rows):
    """A trailing assistant turn becomes the reference: it is stripped from the prompt, surfaced as
    ``labels``, and scored by ``metric='acc'`` (exact string match on the first candidate)."""
    patch_prompt_rows([{
        'messages': P1 + [{'role': 'assistant', 'content': '4'}],
        'solution': '4'
    }])
    infer_config = InferConfig(
        cache_files=[_cache(tmp_path)], num_return_sequences=3, batch_size=1, output_format='all', metric='acc')
    results = run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), backend='no')

    assert results[0]['labels'] == '4'
    assert results[0]['response'] == '4'  # first candidate matches the reference -> acc 1.0


# --- store_format: the shared store writer infer now serialises through ----------------


def test_arrow_format_writes_one_table_not_jsonl(tmp_path, patch_prompt_rows):
    """``store_format='arrow'`` swaps the incremental jsonl writer for :class:`_ArrowWriter`: the whole
    run is serialised once at finish into the same ``save_to_disk`` container ``swift export
    --to_cached_dataset`` writes, at the ``.arrow`` path derived from the jsonl one -- and NO jsonl is
    produced. This is the end-to-end regression for infer's storage change: same rows, new container."""
    from swift.dev.dataset.store import load_dataset_store
    patch_prompt_rows(_rows())
    out_path = str(tmp_path / 'out.jsonl')
    infer_config = InferConfig(
        cache_files=[_cache(tmp_path)], num_return_sequences=3, batch_size=2, output_format='all',
        store_format='arrow')
    results = run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), backend='no', output_path=out_path)

    arrow_path = str(tmp_path / 'out.arrow')
    assert not os.path.exists(out_path)  # the jsonl writer was not used
    assert os.path.isdir(arrow_path)  # an Arrow store is a save_to_disk directory
    ds = load_dataset_store(arrow_path)
    assert len(ds) == len(results) == 2
    assert ds['response'] == [r['response'] for r in results]
    assert ds['responses'] == [r['responses'] for r in results]


def test_arrow_format_honours_store_fields(tmp_path, patch_prompt_rows):
    """``InferConfig.store_fields`` reaches the writer as the column allow-list, so an arrow dump keeps
    only the named columns -- the offline-RL corpus does not want every emit field."""
    from swift.dev.dataset.store import load_dataset_store
    patch_prompt_rows(_rows())
    out_path = str(tmp_path / 'out.jsonl')
    infer_config = InferConfig(
        cache_files=[_cache(tmp_path)], num_return_sequences=3, batch_size=2, output_format='all',
        store_format='arrow', store_fields=['response', 'messages'])
    run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), backend='no', output_path=out_path)

    ds = load_dataset_store(str(tmp_path / 'out.arrow'))
    assert set(ds.column_names) == {'response', 'messages'}  # 'responses' / 'solution' dropped


def test_arrow_format_overrides_dpo_checkpoint_writer(tmp_path, patch_prompt_rows):
    """Writer-selection precedence: a 'dpo' run normally uses the checkpointed jsonl writer, but
    ``store_format='arrow'`` wins -- ``use_checkpoint`` excludes arrow, so the run serialises one Arrow
    table at finish instead of the tmp/resume/state checkpoint files (and forfeits resume, which the
    validate guard and a runtime warning both flag). Pins the ``store_format != 'arrow'`` clause."""
    from swift.dev.dataset.store import load_dataset_store
    patch_prompt_rows(_rows())
    out_path = str(tmp_path / 'dpo.jsonl')
    infer_config = InferConfig(
        cache_files=[_cache(tmp_path)], num_return_sequences=3, n_best_to_keep=1, batch_size=1,
        output_format='dpo', store_format='arrow')
    run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), rlhf_config=RLHFConfig(reward_funcs=[_exact_reward]), backend='no', output_path=out_path)

    assert not os.path.exists(out_path)  # no checkpointed jsonl was published
    assert os.path.isdir(str(tmp_path / 'dpo.arrow'))
    ds = load_dataset_store(str(tmp_path / 'dpo.arrow'))
    assert len(ds) == 2 and 'messages' in ds.column_names  # dpo pairs, one per prompt


def test_jsonl_is_the_default_store_format(tmp_path, patch_prompt_rows):
    """Regression for the default: with ``store_format`` left unset the run still writes jsonl (the arrow
    branch is opt-in), so the storage change did not move the default container out from under callers."""
    patch_prompt_rows(_rows())
    out_path = str(tmp_path / 'out.jsonl')
    assert InferConfig().store_format == 'jsonl'
    infer_config = InferConfig(cache_files=[_cache(tmp_path)], num_return_sequences=3, batch_size=2, output_format='all')
    results = run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), backend='no', output_path=out_path)

    assert os.path.isfile(out_path) and not os.path.isdir(str(tmp_path / 'out.arrow'))
    assert _read_jsonl(out_path) == results


# --- 'dpo' format ---------------------------------------------------------------------


def test_dpo_format_pairs_best_against_worst(tmp_path, patch_prompt_rows):
    """'dpo' + a reward func: the top-scoring candidate is chosen, the lowest is the rejected_response."""
    patch_prompt_rows(_rows())
    infer_config = InferConfig(
        cache_files=[_cache(tmp_path)],
        num_return_sequences=3,
        n_best_to_keep=1,
        batch_size=2,
        output_format='dpo')
    results = run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), rlhf_config=RLHFConfig(reward_funcs=[_exact_reward]), backend='no')

    assert len(results) == 2  # one chosen/rejected pair per prompt (n_best_to_keep=1)
    row = results[0]
    assert row['messages'][-1] == {'role': 'assistant', 'content': '4'}  # the positive
    # scores [1,0,0]: the ranking picks the first 0-scoring candidate as the negative
    assert row['rejected_response'] in ('five', '22')
    assert row['solution'] == '4'
    assert 'ground_truth' not in row  # these rows carry no trailing reference turn
    # the two prompts are distinct groups, so their ids differ
    assert results[0]['id'] != results[1]['id']
    assert results[1]['messages'][-1] == {'role': 'assistant', 'content': 'Paris'}


def test_dpo_format_without_reward_emits_positives_only(tmp_path, patch_prompt_rows):
    """No reward func -> no ranking, so every candidate is a positive with no rejected turn (naming one
    worse would be a fabrication)."""
    patch_prompt_rows(_rows())
    infer_config = InferConfig(
        cache_files=[_cache(tmp_path)], num_return_sequences=3, n_best_to_keep=1, batch_size=2, output_format='dpo')
    results = run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), backend='no')

    assert len(results) == 6  # 3 positives x 2 prompts
    assert all('rejected_response' not in row and 'rejected_messages' not in row for row in results)
    assert {row['messages'][-1]['content'] for row in results[:3]} == {'4', 'five', '22'}


def test_dpo_format_writes_checkpointed_final(tmp_path, patch_prompt_rows):
    """A 'dpo' run with an output_path uses the checkpointed writer and publishes the final jsonl."""
    patch_prompt_rows(_rows())
    out_path = str(tmp_path / 'dpo.jsonl')
    infer_config = InferConfig(
        cache_files=[_cache(tmp_path)], num_return_sequences=3, n_best_to_keep=1, batch_size=1, output_format='dpo')
    run_infer(
        ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
        GenerationConfig(), rlhf_config=RLHFConfig(reward_funcs=[_exact_reward]), backend='no', output_path=out_path)

    rows = _read_jsonl(out_path)
    assert len(rows) == 2
    assert all('messages' in row for row in rows)


# --- guards ---------------------------------------------------------------------------


def test_backend_no_requires_full_cache_coverage(tmp_path, patch_prompt_rows):
    """backend='no' has no sampler, so a prompt the cache does not cover is a hard error, not a silent
    shrink of the group."""
    patch_prompt_rows(_rows())
    # a cache covering only the first prompt
    cache = write_cache_file(str(tmp_path / 'partial.jsonl'), [(P1, ['4', 'five', '22'])])
    infer_config = InferConfig(cache_files=[cache], num_return_sequences=3, batch_size=2, output_format='all')
    with pytest.raises(ValueError, match="requires cache_files"):
        run_infer(
            ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
            GenerationConfig(), backend='no')


def test_empty_dataset_raises(tmp_path, patch_prompt_rows):
    patch_prompt_rows([])
    infer_config = InferConfig(cache_files=[_cache(tmp_path)], num_return_sequences=3, output_format='all')
    with pytest.raises(ValueError, match='empty dataset'):
        run_infer(
            ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
            GenerationConfig(), backend='no')


def test_dpo_format_requires_at_least_two_candidates(tmp_path, patch_prompt_rows):
    """'dpo' needs both a chosen and a rejected candidate, so num_return_sequences < 2 is rejected up
    front rather than producing pairs with no negative."""
    patch_prompt_rows(_rows())
    infer_config = InferConfig(
        cache_files=[_cache(tmp_path)], num_return_sequences=1, output_format='dpo')
    with pytest.raises(ValueError, match='num_return_sequences >= 2'):
        run_infer(
            ModelConfig(task_type='causal_lm'), TemplateConfig(), DatasetConfig(), infer_config,
            GenerationConfig(), backend='no')

# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI parsing / dispatch tests for ``swift infer`` (no model weights, no GPU).

``parse_infer_configs`` turns argv into the atomic Configs and folds the legacy sampling spellings into
their canonical InferConfig fields; ``infer_main`` then dispatches to the interactive REPL or the
dataset pipeline. These tests pin the argv -> Config mapping (the regression surface for legacy
command parameters) and the interactive guards, using a local directory as ``--model`` so the
checkpoint-args restore never reaches the hub.
"""
import os

import pytest

from swift.dev.cli.infer import (_derive_result_path, _guard_interactive, _interactive_dp_width,
                                 infer_main, parse_infer_configs)
from swift.dev.config import DistributedConfig


@pytest.fixture
def fake_model(tmp_path):
    """An existing local directory: ``--model`` pointing here short-circuits the hub snapshot_download
    that ``load_args_default`` would otherwise trigger for a non-existent model id."""
    d = tmp_path / 'model'
    d.mkdir()
    return str(d)


def test_parse_returns_every_config_key(fake_model):
    result = parse_infer_configs(['--model', fake_model])
    for key in ('model_config', 'template_config', 'dataset_config', 'infer_config', 'generation_config',
                'reward_config', 'multi_turn_config', 'rollout_config', 'distributed_config', 'tuner_config',
                'quantize_config', 'plugin_config', 'cli_config', 'runtime_config'):
        assert key in result, f'missing config key {key}'
    # multi_turn_config is the reward_config under a second name (the multi-turn carrier)
    assert result['multi_turn_config'] is result['reward_config']


def test_legacy_num_samples_drives_num_return_sequences(fake_model):
    """--num_samples (the CLI spelling) and num_return_sequences (the field) are one knob, two names."""
    r = parse_infer_configs(['--model', fake_model, '--num_samples', '4'])
    assert r['infer_config'].num_return_sequences == 4
    assert r['cli_config'].num_samples == 4
    # and the reverse spelling syncs back
    r2 = parse_infer_configs(['--model', fake_model, '--num_return_sequences', '6'])
    assert r2['cli_config'].num_samples == 6
    assert r2['infer_config'].num_return_sequences == 6


def test_legacy_pt_backend_alias(fake_model):
    r = parse_infer_configs(['--model', fake_model, '--infer_backend', 'pt'])
    assert r['infer_config'].infer_backend == 'transformers'


def test_legacy_prm_threshold_folds_into_reward_threshold(fake_model):
    r = parse_infer_configs(['--model', fake_model, '--prm_threshold', '0.3'])
    assert r['infer_config'].reward_threshold == 0.3
    # an explicit reward_threshold wins over the legacy spelling
    r2 = parse_infer_configs(['--model', fake_model, '--prm_threshold', '0.3', '--reward_threshold', '0.7'])
    assert r2['infer_config'].reward_threshold == 0.7


def test_legacy_sampling_batch_spellings(fake_model):
    r = parse_infer_configs(['--model', fake_model, '--num_sampling_batch_size', '8', '--num_sampling_batches', '3'])
    assert r['infer_config'].batch_size == 8
    assert r['infer_config'].max_batches == 3


def test_padding_side_defaults_left(fake_model):
    assert parse_infer_configs(['--model', fake_model])['template_config'].padding_side == 'left'
    # an explicit value is respected
    r = parse_infer_configs(['--model', fake_model, '--padding_side', 'right'])
    assert r['template_config'].padding_side == 'right'


def test_stream_derived_from_dataset_presence(fake_model):
    # no dataset -> interactive -> stream on
    assert parse_infer_configs(['--model', fake_model])['generation_config'].stream is True
    # a dataset run -> batch sampling -> stream off
    r = parse_infer_configs(['--model', fake_model, '--dataset', '/tmp/does_not_matter.jsonl'])
    assert r['generation_config'].stream is False


def test_all_format_warns_on_ignored_best_of_n_knobs(fake_model, caplog):
    """output_format='all' stores every candidate as-is, so a best-of-n ranking knob is silently
    ignored; the CLI says so instead of dropping it."""
    import logging
    with caplog.at_level(logging.WARNING, logger='swift.dev'):
        parse_infer_configs(['--model', fake_model, '--output_format', 'all', '--n_best_to_keep', '3'])
    assert any("output_format='all' ignores" in rec.message for rec in caplog.records)


def test_derive_result_path_explicit_wins(fake_model, tmp_path):
    target = str(tmp_path / 'custom.jsonl')
    r = parse_infer_configs(['--model', fake_model, '--result_path', target])
    _derive_result_path(r)
    assert r['infer_config'].result_path == os.path.abspath(target)


def test_derive_result_path_dataset_fallback(fake_model):
    r = parse_infer_configs(['--model', fake_model, '--dataset', '/tmp/x.jsonl'])
    _derive_result_path(r)
    path = r['infer_config'].result_path
    assert path and path.endswith('.jsonl') and os.path.isabs(path)
    assert os.path.join('result', 'model', 'infer_result') in path


def test_derive_result_path_no_dataset_no_path_stays_none(fake_model):
    r = parse_infer_configs(['--model', fake_model])
    _derive_result_path(r)
    assert r['infer_config'].result_path is None


# --- interactive guards ---------------------------------------------------------------


def test_guard_interactive_rejects_pooling_task_type():
    for task_type in ('seq_cls', 'embedding', 'reranker', 'generative_reranker'):
        with pytest.raises(ValueError, match='forward pass'):
            _guard_interactive(None, task_type)


def test_guard_interactive_rejects_multiple_dp_drivers(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError, match='single data-parallel driver'):
        _guard_interactive(None, 'causal_lm')


def test_guard_interactive_allows_single_dp(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '1')
    _guard_interactive(None, 'causal_lm')  # must not raise


def test_interactive_dp_width_local_reads_world_size(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '4')
    assert _interactive_dp_width(None) == 4
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    assert _interactive_dp_width(None) == 1


def test_interactive_dp_width_ray_reads_mesh(monkeypatch):
    """Under mode='ray' the width is the sampler mesh's data world size, not the torchrun world."""
    import swift.dev.builders as builders

    class _Mesh:
        data_world_size = 1

    monkeypatch.setattr(builders, 'build_device_mesh_if_dp', lambda dc: _Mesh())
    assert _interactive_dp_width(DistributedConfig(mode='ray')) == 1
    # ray with dp=1 (tp=N) is a single driver -> the guard allows it
    _guard_interactive(DistributedConfig(mode='ray'), 'causal_lm')


# --- infer_main dispatch --------------------------------------------------------------


def test_infer_main_dispatches_to_dataset_pipeline(fake_model, monkeypatch):
    """A dataset run calls run_infer (not the REPL), threading the parsed configs through."""
    import swift.dev.config as config_mod
    import swift.dev.recipe as recipe_mod

    captured = {}

    def fake_run_infer(*a, **k):
        captured['args'] = (a, k)
        return ['row']

    monkeypatch.setattr(config_mod, 'process_and_validate_configs', lambda *a, **k: None)
    monkeypatch.setattr(recipe_mod, 'run_infer', fake_run_infer)
    monkeypatch.setattr(recipe_mod, 'infer_cli', lambda *a, **k: pytest.fail('REPL must not run for a dataset'))

    out = infer_main(['--model', fake_model, '--dataset', '/tmp/x.jsonl', '--num_samples', '3'])
    assert out == ['row']
    _, kwargs = captured['args']
    assert kwargs['backend'] == 'transformers'
    assert kwargs['rlhf_config'] is not None  # the reward/multi-turn carrier is always threaded


def test_infer_main_dispatches_to_repl_without_dataset(fake_model, monkeypatch):
    import swift.dev.config as config_mod
    import swift.dev.recipe as recipe_mod

    called = {}
    monkeypatch.setattr(config_mod, 'process_and_validate_configs', lambda *a, **k: None)
    monkeypatch.setattr(recipe_mod, 'run_infer', lambda *a, **k: pytest.fail('pipeline must not run interactively'))
    monkeypatch.setattr(recipe_mod, 'infer_cli', lambda *a, **k: called.setdefault('ok', True))
    monkeypatch.setenv('WORLD_SIZE', '1')

    infer_main(['--model', fake_model])  # no dataset -> interactive
    assert called.get('ok') is True

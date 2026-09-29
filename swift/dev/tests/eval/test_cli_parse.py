# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI parsing / dispatch tests for ``swift eval`` (no model weights, no GPU).

``parse_eval_configs`` turns argv into the atomic Configs; ``eval_main`` threads them into ``run_eval``. These
pin the argv -> Config mapping for *every* eval knob, so a renamed field, a lost ``pt -> transformers`` rewrite,
a dropped engine-arg passthrough, or a retired flag that stops being refused is caught here rather than at
eval time. ``--load_args false`` keeps the checkpoint-args restore off the hub, so a placeholder ``--model m``
never triggers a download.
"""
import pytest

from swift.dev.builders import build_engine_args
from swift.dev.cli.eval import eval_main, parse_eval_configs
from swift.dev.config import QuantizeConfig

#: Every parse case carries these so ``load_args_default=True`` never reaches the hub for the fake model.
BASE = ['--model', 'm', '--load_args', 'false']


def _parse(*extra):
    return parse_eval_configs([*BASE, *extra])


# --------------------------------------------------------------------------- eval flags -> EvalConfig
def test_every_eval_flag_lands_on_eval_config(tmp_path):
    out_dir = tmp_path / 'reports'
    jsonl = tmp_path / 'results.jsonl'
    configs = _parse(
        '--eval_dataset', 'gsm8k', 'mmlu',
        '--eval_limit', '5',
        '--eval_num_proc', '8',
        '--eval_generation_config', '{"max_tokens": 16, "temperature": 0.0}',
        '--eval_dataset_args', '{"gsm8k": {"few_shot_num": 2}}',
        '--extra_eval_args', '{"use_cache": false}',
        '--eval_output_dir', str(out_dir),
        '--result_jsonl', str(jsonl),
    )
    eval_config = configs['eval_config']
    assert eval_config.eval_dataset == ['gsm8k', 'mmlu']
    assert eval_config.eval_limit == 5
    assert eval_config.eval_num_proc == 8
    assert eval_config.eval_generation_config == {'max_tokens': 16, 'temperature': 0.0}
    assert eval_config.eval_dataset_args == {'gsm8k': {'few_shot_num': 2}}
    assert eval_config.extra_eval_args == {'use_cache': False}
    assert eval_config.eval_output_dir == str(out_dir)
    assert eval_config.result_jsonl == str(jsonl)


def test_result_jsonl_defaults_to_none():
    assert _parse('--eval_dataset', 'gsm8k')['eval_config'].result_jsonl is None


def test_relative_output_dir_is_absolutized():
    import os
    eval_config = _parse('--eval_dataset', 'gsm8k', '--eval_output_dir', 'rel_out')['eval_config']
    assert os.path.isabs(eval_config.eval_output_dir)
    assert eval_config.eval_output_dir.endswith('rel_out')


# --------------------------------------------------------------------------- backend selection
@pytest.mark.parametrize('backend', ['vllm', 'sglang', 'transformers'])
def test_local_infer_backend_lands_on_infer_config(backend):
    assert _parse('--infer_backend', backend)['infer_config'].sampler == backend


def test_pt_backend_is_rewritten_to_transformers():
    """``pt`` is the legacy spelling; eval normalizes it so ``build_sampler``/``build_engine_args`` see one name."""
    assert _parse('--infer_backend', 'pt')['infer_config'].sampler == 'transformers'


# --------------------------------------------------------------------------- engine args verbatim
def test_vllm_engine_args_reach_the_engine_verbatim():
    configs = _parse('--infer_backend', 'vllm', '--vllm_gpu_memory_utilization', '0.5',
                     '--vllm_tensor_parallel_size', '2')
    engine_args = build_engine_args('vllm', configs['infer_config'], configs['rollout_config'])
    # the vllm_ prefix is stripped, so the engine sees its own flag names
    assert engine_args['gpu_memory_utilization'] == 0.5
    assert engine_args['tensor_parallel_size'] == 2
    # server-only / non-engine keys are never forwarded to an in-process eval engine
    assert 'mode' not in engine_args
    assert 'server_base_url' not in engine_args


def test_transformers_engine_args_carry_max_batch_size():
    configs = _parse('--infer_backend', 'transformers', '--max_batch_size', '4')
    engine_args = build_engine_args('transformers', configs['infer_config'], configs['rollout_config'])
    assert engine_args == {'max_batch_size': 4}


# --------------------------------------------------------------------------- adapters (LoRA live-load)
def test_bare_adapter_path_lands_on_tuner_config(tmp_path):
    adapter = tmp_path / 'checkpoint-100'
    adapter.mkdir()
    configs = _parse('--adapters', str(adapter))
    assert configs['tuner_config'].adapters == [str(adapter)]


def test_multiple_adapters_parse_in_order(tmp_path):
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    first.mkdir()
    second.mkdir()
    configs = _parse('--adapters', str(first), str(second))
    # eval scores a single model, so run_eval uses adapters[0] and warns about the rest (pinned in the e2e).
    assert configs['tuner_config'].adapters == [str(first), str(second)]


def test_quantize_config_is_part_of_the_surface():
    assert isinstance(_parse('--eval_dataset', 'gsm8k')['quantize_config'], QuantizeConfig)


# --------------------------------------------------------------------------- eval_main dispatch
def test_eval_main_threads_backend_engine_args_adapters_and_quantize(monkeypatch, tmp_path):
    """``eval_main`` must hand ``run_eval`` the resolved backend, the verbatim engine args, the adapters and the
    quantize config -- a break in any of these silently changes what the example actually runs."""
    captured = {}
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)

    def fake_run_eval(model_config, template_config, eval_config, **kwargs):
        captured['model_config'] = model_config
        captured['eval_config'] = eval_config
        captured.update(kwargs)
        return {'Native': []}

    monkeypatch.setattr('swift.dev.recipe.run_eval', fake_run_eval)
    adapter = tmp_path / 'lora'
    adapter.mkdir()
    eval_main([
        *BASE, '--infer_backend', 'vllm', '--eval_dataset', 'gsm8k', '--vllm_gpu_memory_utilization', '0.5',
        '--adapters', str(adapter)
    ])
    assert captured['backend'] == 'vllm'
    assert captured['engine_args']['gpu_memory_utilization'] == 0.5
    assert captured['adapters'] == [str(adapter)]
    assert isinstance(captured['quantize_config'], QuantizeConfig)
    assert captured['eval_config'].eval_dataset == ['gsm8k']


def test_eval_main_passes_no_adapters_when_unset(monkeypatch):
    captured = {}
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)
    monkeypatch.setattr(
        'swift.dev.recipe.run_eval',
        lambda model_config, template_config, eval_config, **kwargs: captured.update(kwargs) or {'Native': []})
    eval_main([*BASE, '--infer_backend', 'transformers', '--eval_dataset', 'gsm8k'])
    # no --adapters -> the tuner's empty list, which run_eval folds to "no adapter" via `adapters or None`.
    assert not captured['adapters']
    assert captured['backend'] == 'transformers'


# --------------------------------------------------------------------------- retired flags are refused
@pytest.mark.parametrize('flag,value', [
    ('--eval_url', 'http://127.0.0.1:8000/v1'),
    ('--eval_backend', 'opencompass'),
    ('--local_dataset', 'true'),
    ('--merge_lora', 'true'),
    ('--temperature', '0.7'),
    ('--top_p', '0.9'),
    ('--top_k', '20'),
    ('--max_new_tokens', '128'),
    ('--host', '0.0.0.0'),
])
def test_retired_flags_are_refused_with_a_classified_reason(flag, value):
    """eval is sampler-only/Native-only and starts no server, so its former remote/backend/serving/per-flag
    decoding options must be refused with a reason and an alternative -- not silently ignored, not a bare
    "Unrecognized arguments"."""
    with pytest.raises(ValueError) as exc:
        parse_eval_configs([*BASE, '--eval_dataset', 'gsm8k', flag, value])
    assert f'{flag} is unsupported' in str(exc.value)
    assert 'Alternative:' in str(exc.value)

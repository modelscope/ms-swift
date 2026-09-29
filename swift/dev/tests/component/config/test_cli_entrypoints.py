"""Fast entry-point smoke tests for every public v5 command."""
from __future__ import annotations
import importlib

import pytest

from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    LoggingConfig,
    MegatronConfig,
    ModelConfig,
    MoEConfig,
    QuantizeConfig,
    RLHFConfig,
    TemplateConfig,
    TrainConfig,
)


def _patch_service_runtime(monkeypatch, events):
    # Every command main imports process_and_validate_configs from swift.dev.config and calls it once
    # before dispatching its recipe. bootstrap_run is folded into that single entry, so 'process' is the
    # whole pre-recipe lifecycle a command main owns.
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs',
                        lambda *_args, **_kwargs: events.append('process'))


@pytest.mark.parametrize('command', ['infer', 'deploy', 'eval', 'merge', 'export'])
def test_service_and_artifact_entrypoints_follow_lifecycle(monkeypatch, tmp_path, command):
    events = []
    _patch_service_runtime(monkeypatch, events)

    # --load_args false keeps each parse off the checkpoint-args restore path, which would otherwise try
    # to fetch the fake model id 'm' from the hub.
    if command == 'infer':
        from swift.dev.cli.infer import infer_main

        monkeypatch.setattr('swift.dev.recipe.infer_cli', lambda *_args, **_kwargs: events.append('recipe'))
        infer_main(['--model', 'm', '--load_args', 'false'])
    elif command == 'deploy':
        from swift.dev.cli.deploy import deploy_main

        monkeypatch.setattr('swift.dev.recipe.run_deploy', lambda *_args, **_kwargs: events.append('recipe'))
        deploy_main(['--model', 'm', '--load_args', 'false'])
    elif command == 'eval':
        from swift.dev.cli.eval import eval_main

        monkeypatch.setattr('swift.dev.recipe.run_eval', lambda *_args, **_kwargs: events.append('recipe'))
        eval_main(['--model', 'm', '--eval_dataset', 'bench', '--load_args', 'false'])
    elif command == 'merge':
        from swift.dev.cli.merge import merge_main

        monkeypatch.setattr('swift.dev.recipe.run_merge_lora', lambda *_args, **_kwargs: events.append('recipe'))
        merge_main(['--model', 'm', '--adapters', 'a', '--load_args', 'false'])
    else:
        from swift.dev.cli.export import export_main

        monkeypatch.setattr('swift.dev.recipe.run_to_peft_format', lambda *_args, **_kwargs: events.append('recipe'))
        export_main([
            '--model', 'm', '--to_peft_format', 'true', '--adapters', 'a', '--load_args', 'false', '--output_dir',
            str(tmp_path / 'out')
        ])

    assert events[0] == 'process'
    assert events[-1] == 'recipe'


def test_shared_service_config_lifecycle_processes_before_validation(monkeypatch):
    from swift.dev.cli.infer import parse_infer_configs
    from swift.dev.config import process_and_validate_configs

    events = []
    # process_and_validate_configs resolves process_configs/bootstrap_run as process.py globals and
    # imports validate_configs from .validate at call time, so those are the real patch points.
    monkeypatch.setattr('swift.dev.config.process.process_configs', lambda *_a, **_k: events.append('process'))
    monkeypatch.setattr('swift.dev.config.validate.validate_configs', lambda *_a, **_k: events.append('validate'))
    monkeypatch.setattr('swift.dev.config.process.bootstrap_run', lambda *_a, **_k: events.append('bootstrap'))
    configs = parse_infer_configs(['--model', 'm', '--load_args', 'false'])
    process_and_validate_configs(configs)
    assert events == ['process', 'validate', 'bootstrap']


@pytest.mark.parametrize('parser,flag', [
    ('swift.dev.cli.infer.parse_infer_configs', '--infer_backend'),
    # --sampler_engine is the legacy alias for --infer_backend (the sample command folded into infer);
    # lmdeploy is not a valid backend either way, so argparse rejects it before any parse completes.
    ('swift.dev.cli.infer.parse_infer_configs', '--sampler_engine'),
])
def test_lmdeploy_is_rejected_at_parse_time(parser, flag):
    module_name, function_name = parser.rsplit('.', 1)
    parse = getattr(importlib.import_module(module_name), function_name)
    with pytest.raises(SystemExit):
        parse([flag, 'lmdeploy'])


def _training_configs(task_type='causal_lm', backend=None):
    """A parsed-Config mapping shaped like parse_sft_configs' return (backend selectable)."""
    return {
        'model_config': ModelConfig(model='m', task_type=task_type),
        'plugin_config': None,
        'template_config': TemplateConfig(),
        'dataset_config': DatasetConfig(dataset=['d']),
        'train_config': TrainConfig(),
        'distributed_config': DistributedConfig(backend=backend),
        'checkpoint_config': CheckpointConfig(),
        'logging_config': LoggingConfig(report_to=['none']),
        'tuner_config': None,
        'quantize_config': QuantizeConfig(),
        'megatron_config': MegatronConfig(),
        'moe_config': MoEConfig(),
    }


@pytest.mark.parametrize('task_type,recipe_name', [
    ('causal_lm', 'run_sft'),
    ('seq_cls', 'run_seq_cls'),
    ('embedding', 'run_embedding'),
    ('reranker', 'run_reranker'),
    ('generative_reranker', 'run_reranker'),
])
def test_sft_dispatches_every_supported_task(monkeypatch, task_type, recipe_name):
    from swift.dev.cli import sft as sft_cli

    monkeypatch.setattr(sft_cli, 'parse_sft_configs', lambda _argv: _training_configs(task_type))
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_args, **_kwargs: None)
    expected = object()
    for name in ('run_sft', 'run_seq_cls', 'run_embedding', 'run_reranker'):
        result = expected if name == recipe_name else object()
        monkeypatch.setattr(f'swift.dev.recipe.{name}', lambda *_args, _result=result, **_kwargs: _result)

    assert sft_cli.sft_main([]) is expected


@pytest.mark.parametrize('rlhf_type,recipe_module,recipe_name', [
    ('dpo', 'run_dpo', 'run_dpo'),
    ('kto', 'run_dpo', 'run_dpo'),
    ('cpo', 'run_dpo', 'run_dpo'),
    ('orpo', 'run_dpo', 'run_dpo'),
    ('simpo', 'run_dpo', 'run_dpo'),
    ('rm', 'run_dpo', 'run_dpo'),
    ('grpo', 'run_grpo', 'run_grpo'),
    ('ppo', 'run_ppo', 'run_ppo'),
    ('gkd', 'run_gkd', 'run_gkd'),
])
def test_rlhf_dispatches_every_supported_algorithm(monkeypatch, rlhf_type, recipe_module, recipe_name):
    from swift.dev.recipe.run_rlhf import run_rlhf

    expected = object()
    recipe = importlib.import_module(f'swift.dev.recipe.{recipe_module}')
    monkeypatch.setattr(recipe, recipe_name, lambda *_args, **_kwargs: expected)
    result = run_rlhf(
        ModelConfig(),
        TemplateConfig(),
        DatasetConfig(),
        TrainConfig(),
        DistributedConfig(),
        CheckpointConfig(),
        object(),
        RLHFConfig(rlhf_type=rlhf_type),
    )
    assert result is expected


@pytest.mark.parametrize('backend,expects_megatron', [(None, False), ('megatron', True)])
def test_pt_entrypoint_enforces_pretraining_contract(monkeypatch, backend, expects_megatron):
    from swift.dev.cli import pt as pt_cli

    configs = _training_configs(backend=backend)
    monkeypatch.setattr('swift.dev.cli.sft.parse_sft_configs', lambda _argv, command='pt': configs)
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_args, **_kwargs: None)
    captured = {}

    def fake_run_pt(*_args, **kwargs):
        captured.update(kwargs)
        return 'pt'

    monkeypatch.setattr('swift.dev.recipe.run_pt', fake_run_pt)
    assert pt_cli.pt_main([]) == 'pt'
    assert configs['model_config'].task_type == 'causal_lm'
    assert configs['template_config'].use_chat_template is False
    assert configs['template_config'].loss_scale == 'all'
    # The Megatron Configs reach run_pt only on the megatron backend; the HF task recipes do not take them.
    assert ('megatron_config' in captured) is expects_megatron


def test_infer_writer_appends_existing_results(tmp_path):
    import json

    from swift.dev.recipe.run_infer import _IncrementalWriter

    path = tmp_path / 'results.jsonl'
    path.write_text('{"response": "old"}\n', encoding='utf-8')
    writer = _IncrementalWriter(str(path), batch_size=1)
    writer.write([{'response': 'new'}])
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert rows == [{'response': 'old'}, {'response': 'new'}]


def test_bootstrap_seeds_each_rank(monkeypatch, tmp_path):
    from swift.dev.config import bootstrap_run

    seeds = []
    monkeypatch.setenv('RANK', '3')
    monkeypatch.setattr('swift.utils.seed_everything', seeds.append)
    bootstrap_run(ModelConfig(), CheckpointConfig(output_dir=str(tmp_path)), seed=7, resolve_model=False)
    assert seeds == [10]


def test_export_merge_then_quantize_chains_model(monkeypatch, tmp_path):
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    configs = parse_export_configs([
        '--model', 'm', '--adapters', 'a', '--merge_lora', 'true', '--quant_method', 'gptq', '--quant_bits', '4',
        '--dataset', 'd', '--load_args', 'false', '--output_dir', str(tmp_path / 'out'),
    ])
    # run_export_configs imports process_and_validate_configs from swift.dev.config and calls it with
    # add_version/create_output_dir kwargs; --load_args false keeps parse off the checkpoint-args restore
    # path, which would otherwise try to fetch the fake model id 'm' from the hub.
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.recipe.run_merge_lora', lambda *_args, **_kwargs: 'merged')

    def quantize(model_config, *_args, **_kwargs):
        assert model_config.model == 'merged'
        return 'quantized'

    monkeypatch.setattr('swift.dev.recipe.run_quantize', quantize)
    assert run_export_configs(configs) == 'quantized'
    assert configs['tuner_config'].adapters == []


def test_export_to_cached_dataset_forwards_store_flags(monkeypatch, tmp_path):
    """``--to_cached_dataset`` dispatches to ``export_cached_dataset`` and threads every store knob through
    verbatim: the CLI flags land on the recipe (template_mode / store_format / store_encoded /
    store_fields), and the returned train_dir becomes the command's result. This pins the CLI->recipe
    wiring for the unified store so a renamed or dropped flag is caught here. ``--load_args false`` keeps
    the parse off the checkpoint-args restoration path, which would otherwise try to fetch model 'm'."""
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    configs = parse_export_configs([
        '--model', 'm', '--dataset', 'd', '--to_cached_dataset', 'true', '--template_mode', 'rlhf',
        '--store_format', 'jsonl', '--store_encoded', 'true', '--store_fields', 'input_ids', 'labels',
        '--load_args', 'false', '--output_dir', str(tmp_path / 'out'),
    ])
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)

    captured = {}

    def fake_export(model_config, template_config, dataset_config, **kwargs):
        captured.update(kwargs)
        return str(tmp_path / 'out' / 'train'), None

    monkeypatch.setattr('swift.dev.recipe.export_cached_dataset', fake_export)
    assert run_export_configs(configs) == str(tmp_path / 'out' / 'train')
    assert captured['template_mode'] == 'rlhf'
    assert captured['store_format'] == 'jsonl'
    assert captured['store_encoded'] is True
    assert captured['store_fields'] == ['input_ids', 'labels']
    assert captured['output_dir'] == str(tmp_path / 'out')


def test_export_to_mcore_dispatches_run_convert(monkeypatch, tmp_path):
    """``--to_mcore`` dispatches to ``run_convert`` with the parsed ConvertConfig (direction flag set) and
    the resolved output_dir, and the recipe's return becomes the command's result. Pins the CLI->recipe
    wiring for HF->mcore. ``--load_args false`` keeps the parse off the checkpoint-args restore path,
    which would otherwise try to fetch the fake model id 'm' from the hub."""
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    configs = parse_export_configs([
        '--model', 'm', '--to_mcore', 'true', '--load_args', 'false', '--output_dir',
        str(tmp_path / 'mcore'),
    ])
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)

    captured = {}

    def fake_convert(model_config, convert_config, **kwargs):
        captured['convert_config'] = convert_config
        captured['output_dir'] = kwargs.get('output_dir')
        return str(tmp_path / 'mcore')

    monkeypatch.setattr('swift.dev.recipe.run_convert', fake_convert)
    assert run_export_configs(configs) == str(tmp_path / 'mcore')
    assert captured['convert_config'].to_mcore is True
    assert captured['convert_config'].to_hf is False
    assert captured['output_dir'] == str(tmp_path / 'mcore')


def test_export_to_hf_dispatches_run_convert(monkeypatch, tmp_path):
    """``--to_hf`` (with ``--mcore_model`` as the source) dispatches to ``run_convert`` carrying both the
    direction flag and the mcore source path. ``--safe_serialization true`` is required by the to_hf
    guard, so it is passed explicitly here."""
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    configs = parse_export_configs([
        '--model', 'm', '--mcore_model', '/src/mcore', '--to_hf', 'true', '--safe_serialization', 'true',
        '--load_args', 'false', '--output_dir',
        str(tmp_path / 'hf'),
    ])
    monkeypatch.setattr('swift.dev.config.process_and_validate_configs', lambda *_a, **_k: None)

    captured = {}

    def fake_convert(model_config, convert_config, **kwargs):
        captured['convert_config'] = convert_config
        return str(tmp_path / 'hf')

    monkeypatch.setattr('swift.dev.recipe.run_convert', fake_convert)
    assert run_export_configs(configs) == str(tmp_path / 'hf')
    assert captured['convert_config'].to_hf is True
    assert captured['convert_config'].mcore_model == '/src/mcore'


def test_export_rejects_merge_lora_with_to_mcore(tmp_path):
    """``--merge_lora`` and ``--to_mcore``/``--to_hf`` are separate commands in the dev export CLI; the
    combination is refused by ``_validate_export`` before any recipe runs."""
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    configs = parse_export_configs([
        '--model', 'm', '--merge_lora', 'true', '--to_mcore', 'true', '--load_args', 'false',
    ])
    with pytest.raises(ValueError, match='cannot be combined'):
        run_export_configs(configs)


def test_export_rejects_both_convert_directions(tmp_path):
    """Setting both ``--to_mcore`` and ``--to_hf`` is ambiguous and refused at the CLI validate step."""
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    configs = parse_export_configs([
        '--model', 'm', '--to_mcore', 'true', '--to_hf', 'true', '--load_args', 'false',
    ])
    with pytest.raises(ValueError, match='exactly one conversion direction'):
        run_export_configs(configs)


def test_export_to_hf_requires_safe_serialization(tmp_path):
    """mcore->HF always writes safetensors, so ``--safe_serialization false`` is refused rather than
    silently producing a format the bridge cannot write."""
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    configs = parse_export_configs([
        '--model', 'm', '--mcore_model', '/src/mcore', '--to_hf', 'true', '--safe_serialization', 'false',
        '--load_args', 'false',
    ])
    with pytest.raises(NotImplementedError, match='always writes safetensors'):
        run_export_configs(configs)

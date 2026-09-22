"""Fast entry-point smoke tests for every public v5 command."""
from __future__ import annotations
import asyncio
import importlib
import sys
from dataclasses import dataclass
from types import ModuleType

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
    monkeypatch.setattr('swift.dev.cli.runtime.process_and_validate_configs', lambda _configs: events.append('process'))
    monkeypatch.setattr('swift.dev.cli.runtime.bootstrap_run', lambda *_args, **_kwargs: events.append('bootstrap'))


@pytest.mark.parametrize('command', ['infer', 'deploy', 'rollout', 'sample', 'eval', 'app', 'merge_lora', 'export'])
def test_service_and_artifact_entrypoints_follow_lifecycle(monkeypatch, tmp_path, command):
    events = []
    _patch_service_runtime(monkeypatch, events)

    if command == 'infer':
        from swift.dev.cli.infer import infer_main

        monkeypatch.setattr('swift.dev.recipe.infer_cli', lambda *_args, **_kwargs: events.append('recipe'))
        infer_main(['--model', 'm'])
    elif command == 'deploy':
        from swift.dev.cli.deploy import deploy_main

        monkeypatch.setattr('swift.dev.recipe.run_deploy', lambda *_args, **_kwargs: events.append('recipe'))
        deploy_main(['--model', 'm'])
    elif command == 'rollout':
        from swift.dev.cli.rollout import rollout_main

        monkeypatch.setattr('swift.dev.recipe.run_rollout', lambda *_args, **_kwargs: events.append('recipe'))
        rollout_main(['--model', 'm'])
    elif command == 'sample':
        from swift.dev.cli.sample import sample_main

        monkeypatch.setattr('swift.dev.recipe.run_sampling', lambda *_args, **_kwargs: events.append('recipe'))
        sample_main(['--model', 'm', '--dataset', 'd'])
    elif command == 'eval':
        from swift.dev.cli.eval import eval_main

        monkeypatch.setattr('swift.dev.recipe.run_eval', lambda *_args, **_kwargs: events.append('recipe'))
        eval_main(['--model', 'm', '--eval_url', 'http://localhost:8000'])
    elif command == 'app':
        from swift.dev.cli import app as app_cli

        monkeypatch.setattr(app_cli, '_derive_ui_defaults', lambda _configs: events.append('derive'))
        monkeypatch.setattr('swift.dev.recipe.run_app', lambda *_args, **_kwargs: events.append('recipe'))
        app_cli.app_main(['--model', 'm', '--base_url', 'http://localhost:8000'])
    elif command == 'merge_lora':
        from swift.dev.cli.merge_lora import merge_lora_main

        monkeypatch.setattr('swift.dev.recipe.run_merge_lora', lambda *_args, **_kwargs: events.append('recipe'))
        merge_lora_main(['--model', 'm', '--adapters', 'a'])
    else:
        from swift.dev.cli.export import export_main

        monkeypatch.setattr('swift.dev.recipe.run_export_ollama', lambda *_args, **_kwargs: events.append('recipe'))
        export_main(['--model', 'm', '--to_ollama', 'true', '--output_dir', str(tmp_path / 'out')])

    expected_prefix = ['process', 'derive', 'bootstrap'] if command == 'app' else ['process', 'bootstrap']
    assert events[:len(expected_prefix)] == expected_prefix
    assert events[-1] == 'recipe'


def test_shared_service_config_lifecycle_processes_before_validation(monkeypatch):
    from swift.dev.cli.infer import parse_infer_configs
    from swift.dev.cli.runtime import process_and_validate_configs

    events = []
    monkeypatch.setattr('swift.dev.config.process_configs', lambda *_args, **_kwargs: events.append('process'))
    monkeypatch.setattr('swift.dev.config.validate_configs', lambda *_args, **_kwargs: events.append('validate'))
    process_and_validate_configs(parse_infer_configs(['--model', 'm']))
    assert events == ['process', 'validate']


@pytest.mark.parametrize('parser,flag', [
    ('swift.dev.cli.infer.parse_infer_configs', '--infer_backend'),
    ('swift.dev.cli.sample.parse_sample_configs', '--sampler_engine'),
])
def test_lmdeploy_is_rejected_at_parse_time(parser, flag):
    module_name, function_name = parser.rsplit('.', 1)
    parse = getattr(importlib.import_module(module_name), function_name)
    with pytest.raises(SystemExit):
        parse([flag, 'lmdeploy'])


def _training_configs(task_type='causal_lm'):
    return (
        ModelConfig(model='m', task_type=task_type),
        TemplateConfig(),
        DatasetConfig(dataset=['d']),
        TrainConfig(),
        DistributedConfig(),
        CheckpointConfig(),
        LoggingConfig(report_to=['none']),
        None,
        QuantizeConfig(),
    )


def _megatron_training_configs(task_type='causal_lm'):
    return (*_training_configs(task_type), MegatronConfig(), MoEConfig())


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
    monkeypatch.setattr('swift.dev.config.process_configs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.config.validate_configs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.cli.runtime.bootstrap_run', lambda *_args, **_kwargs: None)
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


@pytest.mark.parametrize('module_name,parser_path', [
    ('swift.dev.cli.pt', 'swift.dev.cli.sft.parse_sft_configs'),
    ('swift.dev.cli.megatron_pt', 'swift.dev.cli.megatron.parse_megatron_configs'),
])
def test_pt_entrypoints_enforce_pretraining_contract(monkeypatch, module_name, parser_path):
    module = importlib.import_module(module_name)
    configs = _megatron_training_configs() if module_name.endswith('megatron_pt') else _training_configs()
    monkeypatch.setattr(parser_path, lambda _argv: configs)
    monkeypatch.setattr('swift.dev.config.process_configs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.config.validate_configs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.cli.runtime.bootstrap_run', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.recipe.run_pt', lambda *_args, **_kwargs: 'pt')

    entry = module.megatron_pt_main if module_name.endswith('megatron_pt') else module.pt_main
    assert entry([]) == 'pt'
    assert configs[0].task_type == 'causal_lm'
    assert configs[1].use_chat_template is False
    assert configs[1].loss_scale == 'all'


def test_megatron_wrappers_delegate_to_dev_entrypoints(monkeypatch):
    from swift.dev.cli import megatron_export, megatron_rlhf

    sentinel = object()
    monkeypatch.setattr('swift.dev.cli.rlhf.parse_rlhf_configs', lambda argv, megatron: (argv, megatron))
    monkeypatch.setattr('swift.dev.cli.rlhf.run_rlhf_configs', lambda configs: configs)
    assert megatron_rlhf.megatron_rlhf_main(['--model', 'm']) == (['--model', 'm'], True)

    configs = {'distributed_config': DistributedConfig()}
    monkeypatch.setattr('swift.dev.cli.export.parse_export_configs', lambda _argv: configs)
    monkeypatch.setattr('swift.dev.cli.export.run_export_configs', lambda value: sentinel if value is configs else None)
    assert megatron_export.megatron_export_main([]) is sentinel
    assert configs['distributed_config'].backend == 'megatron'

    from swift.dev.cli import megatron

    training = _megatron_training_configs()
    monkeypatch.setattr(megatron, 'parse_megatron_configs', lambda _argv: training)
    monkeypatch.setattr('swift.dev.config.process_configs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.config.validate_configs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.cli.runtime.bootstrap_run', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.recipe.run_sft', lambda *_args, **_kwargs: sentinel)
    assert megatron.megatron_sft_main([]) is sentinel


def test_rollout_app_protocol_and_shutdown(monkeypatch):
    from swift.dev.config import GenerationConfig, RolloutConfig
    from swift.dev.recipe.run_rollout import build_rollout_app

    @dataclass
    class Params:
        temperature: float = 0.5

    @dataclass
    class Sample:
        decoded: str = 'ok'

    class Sampler:

        def reset_prefix_cache(self):
            return {'reset': True}

    class Engine:

        def __init__(self, *_args, **_kwargs):
            self.sampler = Sampler()
            self.closed = False

        def generate(self, prompts, **_kwargs):
            assert prompts == [[{'role': 'user', 'content': 'hi'}]]
            return [Sample()]

        def shutdown(self):
            self.closed = True

    monkeypatch.setattr('swift.model.get_model_processor', lambda *_args, **_kwargs: (None, object()))
    monkeypatch.setattr('swift.dev.builders.build_template', lambda *_args, **_kwargs: object())
    monkeypatch.setattr('swift.dev.builders.to_sampling_params', lambda *_args, **_kwargs: Params())
    monkeypatch.setattr('swift.dev.rollout.RolloutEngine', Engine)
    rollout = RolloutConfig(vllm_tensor_parallel_size=2, vllm_data_parallel_size=3)
    app = build_rollout_app(ModelConfig(model='m'), TemplateConfig(), rollout, RLHFConfig(), GenerationConfig())
    routes = {route.path: route.endpoint for route in app.routes}

    assert asyncio.run(routes['/health/']()) == {'status': 'ok'}
    assert asyncio.run(routes['/get_world_size/']()) == {'world_size': 6}
    assert asyncio.run(routes['/infer/']({'prompts': [[{'role': 'user', 'content': 'hi'}]]})) == [{'decoded': 'ok'}]
    assert asyncio.run(routes['/reset_prefix_cache/']({})) == {'reset': True}
    asyncio.run(app.router.on_shutdown[0]())
    assert app.state.rollout_engine.closed


def test_eval_remote_service_lifecycle(monkeypatch, tmp_path):
    from swift.dev.config import DeployConfig, EvalConfig, GenerationConfig, InferConfig, RolloutConfig
    from swift.dev.recipe import run_eval as run_eval_recipe

    run_eval_module = importlib.import_module('swift.dev.recipe.run_eval')

    calls = []
    evalscope = ModuleType('evalscope')
    evalscope_run = ModuleType('evalscope.run')
    evalscope_run.run_task = lambda task_cfg: calls.append(task_cfg)
    monkeypatch.setitem(sys.modules, 'evalscope', evalscope)
    monkeypatch.setitem(sys.modules, 'evalscope.run', evalscope_run)
    monkeypatch.setattr(run_eval_module, '_validate_eval_datasets', lambda _config: None)
    monkeypatch.setattr(run_eval_module, 'build_eval_task', lambda *_args: 'task')
    monkeypatch.setattr(run_eval_module, '_summarize', lambda *_args: {'score': 1})

    report = run_eval_recipe(
        ModelConfig(model='m'),
        TemplateConfig(),
        GenerationConfig(),
        InferConfig(),
        RolloutConfig(),
        DeployConfig(),
        EvalConfig(eval_dataset=['bench'], eval_url='http://localhost:8000', eval_output_dir=str(tmp_path)),
    )
    assert calls == ['task']
    assert report['Native'] == {'score': 1}


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
    from swift.dev.cli.runtime import bootstrap_run

    seeds = []
    monkeypatch.setenv('RANK', '3')
    monkeypatch.setattr('swift.utils.seed_everything', seeds.append)
    bootstrap_run(ModelConfig(), CheckpointConfig(output_dir=str(tmp_path)), seed=7, resolve_model=False)
    assert seeds == [10]


def test_export_merge_then_quantize_chains_model(monkeypatch, tmp_path):
    from swift.dev.cli.export import parse_export_configs, run_export_configs

    configs = parse_export_configs([
        '--model', 'm', '--adapters', 'a', '--merge_lora', 'true', '--quant_method', 'gptq', '--quant_bits', '4',
        '--dataset', 'd', '--output_dir', str(tmp_path / 'out'),
    ])
    monkeypatch.setattr('swift.dev.cli.runtime.process_and_validate_configs', lambda _configs: None)
    monkeypatch.setattr('swift.dev.cli.runtime.bootstrap_run', lambda *_args, **_kwargs: None)
    monkeypatch.setattr('swift.dev.recipe.run_merge_lora', lambda *_args, **_kwargs: 'merged')

    def quantize(model_config, *_args, **_kwargs):
        assert model_config.model == 'merged'
        return 'quantized'

    monkeypatch.setattr('swift.dev.recipe.run_quantize', quantize)
    assert run_export_configs(configs) == 'quantized'
    assert configs['tuner_config'].adapters == []

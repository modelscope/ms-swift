"""Fast tests for dev CLI self-parsing (argv -> atomic Configs)."""
from __future__ import annotations
import dataclasses

import pytest

from swift.cli.main import ROUTE_MAPPING, cli_main, resolve_route
from swift.dev.cli.export import parse_export_configs
from swift.dev.cli.runtime import bootstrap_run
from swift.dev.cli.sft import LEGACY_ONLY_CLASSIFICATION, parse_sft_configs
from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    QuantizeConfig,
    RLHFConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
    process_configs,
)
from swift.dev.recipe.assembly import TrainAssembly, _resolve_step_interval


def test_defaults_come_from_configs():
    _, template, _, train, _, checkpoint, logging, tuner, quantize = parse_sft_configs(
        ['--model', 'm', '--dataset', 'd', '--tuner_type', 'lora'])
    assert template.padding_side == TemplateConfig().padding_side == 'right'
    assert train.warmup_ratio == TrainConfig().warmup_ratio == 0.0
    assert train.optim == TrainConfig().optim == 'adamw_torch_fused'
    assert checkpoint.output_dir == CheckpointConfig().output_dir == 'output'
    assert logging.logging_steps == LoggingConfig().logging_steps == 5
    assert tuner.lora_rank == TunerConfig().lora_rank
    assert quantize == QuantizeConfig()


def test_explicit_values_land_in_owning_configs():
    model, template, dataset, train, dist, checkpoint, logging, tuner, quantize = parse_sft_configs([
        '--model', 'm', '--dataset', 'd1', 'd2', '--val_dataset', 'v', '--torch_dtype', 'bfloat16',
        '--padding_side', 'left', '--max_length', '256', '--learning_rate', '0.0001', '--save_steps', '500',
        '--eval_steps', '100', '--logging_steps', '2', '--cp_comm_type', 'p2p', '--tuner_type', 'lora', '--lora_rank',
        '16',
        '--lora_alpha', '64', '--target_modules', 'q_proj',
    ])
    assert isinstance(model, ModelConfig)
    assert isinstance(template, TemplateConfig)
    assert isinstance(dataset, DatasetConfig)
    assert isinstance(train, TrainConfig)
    assert isinstance(dist, DistributedConfig)
    assert isinstance(checkpoint, CheckpointConfig)
    assert isinstance(tuner, TunerConfig)
    assert model.torch_dtype == 'bfloat16'
    assert template.padding_side == 'left' and template.max_length == 256
    assert dataset.dataset == ['d1', 'd2'] and dataset.val_dataset == ['v']
    assert train.learning_rate == 1e-4 and train.eval_steps == 100.0
    assert checkpoint.save_steps == 500.0
    assert logging.logging_steps == 2
    assert dist.cp_comm_type == 'p2p'
    assert tuner.lora_rank == 16 and tuner.lora_alpha == 64
    assert quantize.quant_method is None


def test_mapping_annotations_parse_stably_after_other_typing_imports():
    import swift.rollout.gym_env  # noqa: F401 -- reproduces typing Union cache reordering

    model, _, _, train, dist, *_ = parse_sft_configs([
        '--model', 'm', '--dataset', 'd', '--model_kwargs', '{"x": 1}', '--liger_kernel_config',
        '{"fused_linear_cross_entropy": true}', '--fsdp_config', '{"limit_all_gathers": true}',
    ])
    process_configs(model, TemplateConfig(), DatasetConfig(), train, dist)
    assert model.model_kwargs == {'x': 1}
    assert train.liger_kernel_config == {'fused_linear_cross_entropy': True}
    assert dist.fsdp_config == {'limit_all_gathers': True}


def test_legacy_precision_aliases_map_to_torch_dtype():
    model, *_ = parse_sft_configs(['--model', 'm', '--dataset', 'd', '--bf16', 'true'])
    assert model.torch_dtype == 'bfloat16'
    with pytest.raises(ValueError, match='Conflicting values for --torch_dtype'):
        parse_sft_configs(['--model', 'm', '--dataset', 'd', '--bf16', 'true', '--fp16', 'true'])


def test_full_training_yields_no_tuner_config():
    *_, tuner, quantize = parse_sft_configs(['--model', 'm', '--dataset', 'd', '--tuner_type', 'full'])
    assert tuner is None
    assert quantize.quant_method is None


def test_fractional_intervals_survive_parse_for_step_planning():
    _, _, _, train, _, checkpoint, _, _, _ = parse_sft_configs(
        ['--model', 'm', '--dataset', 'd', '--save_steps', '0.25', '--eval_steps', '0.1'])
    assert checkpoint.save_steps == 0.25
    assert train.eval_steps == 0.1


def test_unknown_and_unwired_flags_fail_loudly():
    with pytest.raises(ValueError, match='Unrecognized'):
        parse_sft_configs(['--model', 'm', '--dataset', 'd', '--definitely_unknown', 'x'])
    with pytest.raises(NotImplementedError, match='optimizer'):
        parse_sft_configs(['--model', 'm', '--dataset', 'd', '--optimizer', 'muon'])
    *_, tuner, _ = parse_sft_configs(['--model', 'm', '--dataset', 'd', '--tuner_type', 'vera'])
    assert tuner.tuner_type == 'vera'


def test_megatron_aliases_work_on_transformers_backend():
    *_, train, _, _, _, _, _ = parse_sft_configs(['--model', 'm', '--dataset', 'd', '--lr', '0.001'])
    assert train.learning_rate == 0.001


@pytest.mark.parametrize('argv', [
    ['--lr', '0.001', '--learning_rate=0.001'],
    ['--lr=1', '--learning-rate', '1.0'],
])
def test_aliases_accept_space_equals_hyphens_and_equal_double_writes(argv):
    *_, train, _, _, _, _, _ = parse_sft_configs(['--model=m', '--dataset', 'd', *argv])
    assert train.learning_rate == pytest.approx(float(argv[1] if argv[0] == '--lr' else argv[0].split('=', 1)[1]))
    assert 'learning_rate' in train._explicit_fields


def test_aliases_reject_conflicting_double_writes():
    with pytest.raises(ValueError, match='Conflicting values for --learning_rate'):
        parse_sft_configs(['--model', 'm', '--dataset', 'd', '--lr=0.001', '--learning-rate', '0.002'])
    with pytest.raises(ValueError, match='Conflicting values for --torch_dtype'):
        parse_sft_configs(['--model', 'm', '--dataset', 'd', '--bf16=1', '--torch-dtype', 'float16'])


def test_false_precision_alias_does_not_override_canonical_dtype():
    model, *_ = parse_sft_configs(
        ['--model', 'm', '--dataset', 'd', '--bf16=0', '--torch_dtype', 'float16'])
    assert model.torch_dtype == 'float16'


def test_legacy_only_gap_is_exhaustively_classified():
    from swift.arguments import SftArguments
    from swift.dev.config import LoggingConfig, QuantizeConfig

    classes = [ModelConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig,
               TunerConfig, LoggingConfig, QuantizeConfig]
    config_fields = {field.name for cls in classes for field in dataclasses.fields(cls)}
    legacy_only = {field.name for field in dataclasses.fields(SftArguments)} - config_fields
    classified = {name for names in LEGACY_ONLY_CLASSIFICATION.values() for name in names} | {'bf16', 'fp16'}
    assert legacy_only == classified


def test_unconsumed_existing_configs_fail_loudly():
    *_, logging, _, quantize = parse_sft_configs(['--model', 'm', '--dataset', 'd', '--report_to', 'wandb'])
    assert logging.report_to == ['wandb']
    *_, quantize = parse_sft_configs(
        ['--model', 'm', '--dataset', 'd', '--tuner_type', 'lora', '--quant_method', 'bnb', '--quant_bits', '4'])
    assert quantize.quant_method == 'bnb' and quantize.quant_bits == 4
    with pytest.raises(ValueError, match='no generation phase'):
        parse_sft_configs(['--model', 'm', '--dataset', 'd', '--top_p', '0.9'])


def test_sft_config_field_names_do_not_collide():
    classes = [
        ModelConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig, LoggingConfig,
        TunerConfig, QuantizeConfig
    ]
    owners = {}
    for cls in classes:
        for field in dataclasses.fields(cls):
            assert field.name not in owners, f'{field.name} belongs to both {owners[field.name]} and {cls.__name__}'
            owners[field.name] = cls.__name__


def test_fractional_intervals_resolve_after_total_steps_are_known():
    assert _resolve_step_interval(0.25, 10, 'save_steps') == 3
    assert _resolve_step_interval(500.0, 1000, 'save_steps') == 500
    assert _resolve_step_interval(None, 10, 'eval_steps') is None
    with pytest.raises(ValueError, match='integer step interval'):
        _resolve_step_interval(1.5, 10, 'save_steps')


def test_prepare_processes_before_validation(monkeypatch):
    events = []
    monkeypatch.setattr('swift.dev.plugin.PluginRegistry.load_configured', lambda *_: events.append('plugins'))
    monkeypatch.setattr('swift.dev.config.process_configs', lambda *_args, **_kwargs: events.append('process'))
    monkeypatch.setattr('swift.dev.config.validate_configs', lambda *_args, **_kwargs: events.append('validate'))
    assembly = TrainAssembly(
        'test', ModelConfig(model='m'), TemplateConfig(), DatasetConfig(), TrainConfig(), DistributedConfig(),
        CheckpointConfig(), logging_config=LoggingConfig())
    assert assembly.prepare() is assembly
    assert events == ['plugins', 'process', 'validate']


def test_logging_steps_reaches_training_loop(monkeypatch):
    captured = {}

    class FakeLoop:

        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr('swift.dev.recipe.train_loop.SFTLoop', FakeLoop)
    monkeypatch.setattr('swift.dev.optimizer.resolve_max_grad_norm', lambda _: 1.0)
    assembly = TrainAssembly(
        'test', ModelConfig(model='m'), TemplateConfig(), DatasetConfig(), TrainConfig(), DistributedConfig(),
        CheckpointConfig(save_only_model=True), logging_config=LoggingConfig(logging_steps=7))
    assembly.model = object()
    assembly.dataloader = []
    assembly.total_opt_steps = 1
    assembly.build_loop()
    assert captured['logging_config'].logging_steps == 7
    assert captured['no_save_optim'] is True
    assert captured['no_save_rng'] is True


def test_model_only_resume_without_trainer_state_starts_from_zero(tmp_path):
    checkpoint = CheckpointConfig(resume_from_checkpoint=str(tmp_path), resume_only_model=True)
    assembly = TrainAssembly(
        'test', ModelConfig(model='m'), TemplateConfig(), DatasetConfig(), TrainConfig(gradient_accumulation_steps=3),
        DistributedConfig(), checkpoint, tuner_config=TunerConfig(tuner_type='lora'))

    class Model:

        def __init__(self):
            self.calls = []

        def load(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    assembly.model = Model()
    state = assembly.resume_model()
    assert state == {'cur_step': 0, 'consumed_train_samples': 0, 'gradient_accumulation_steps': 3}
    assert assembly.model.calls == [((str(tmp_path), ), {'adapter_name': 'default'})]


def test_full_resume_refuses_model_only_checkpoint(tmp_path):
    checkpoint = CheckpointConfig(resume_from_checkpoint=str(tmp_path), resume_only_model=False)
    assembly = TrainAssembly(
        'test', ModelConfig(model='m'), TemplateConfig(), DatasetConfig(), TrainConfig(), DistributedConfig(), checkpoint)
    assembly.model = object()
    with pytest.raises(FileNotFoundError, match='resume_only_model'):
        assembly.resume_model()


def test_use_ray_is_wired_for_megatron_and_refused_for_transformers():
    common = (ModelConfig(), TemplateConfig(), DatasetConfig(), TrainConfig())
    megatron = DistributedConfig(backend='megatron', use_ray=True)
    process_configs(*common, megatron)
    assert megatron.mode == 'ray'

    transformers = DistributedConfig(backend='hf', use_ray=True)
    with pytest.raises(NotImplementedError, match='only wired for the Megatron backend'):
        process_configs(*common, transformers)


def test_gym_config_derives_scheduler_and_validates_multi_turn():
    from swift.dev.config import validate_configs

    common = (ModelConfig(), TemplateConfig(), DatasetConfig(), TrainConfig(), DistributedConfig())
    rlhf = RLHFConfig(rlhf_type='grpo', gym_env='math_env')
    process_configs(*common, rlhf_config=rlhf)
    assert rlhf.use_gym_env is True
    assert rlhf.multi_turn_scheduler == 'gym_scheduler'
    validate_configs(*common, rlhf_config=rlhf)

    rlhf.teacher_model_server = 'http://teacher'
    with pytest.raises(ValueError, match='not supported with multi-turn'):
        validate_configs(*common, rlhf_config=rlhf)


def test_multi_turn_config_rejects_unknown_or_mismatched_scheduler():
    from swift.dev.config.validate import _check_multi_turn

    with pytest.raises(ValueError, match='Unknown multi_turn_scheduler'):
        _check_multi_turn(RLHFConfig(rlhf_type='grpo', multi_turn_scheduler='missing'))
    with pytest.raises(ValueError, match='GYMScheduler-compatible'):
        _check_multi_turn(
            RLHFConfig(rlhf_type='grpo', multi_turn_scheduler='math_tip_trick', use_gym_env=True))
    with pytest.raises(ValueError, match='requires use_gym_env=True'):
        _check_multi_turn(RLHFConfig(rlhf_type='grpo', multi_turn_scheduler='gym_scheduler', gym_env='math_env'))


def test_bootstrap_versions_output_and_abspaths_resume(tmp_path):
    resume = tmp_path / 'checkpoint-1'
    resume.mkdir()
    checkpoint = CheckpointConfig(output_dir=str(tmp_path / 'run'), resume_from_checkpoint=str(resume))
    model = ModelConfig(model_kwargs='{"cli_runtime_test": "yes"}')
    bootstrap_run(model, checkpoint)
    assert checkpoint.output_dir.startswith(str((tmp_path / 'run').resolve()))
    assert checkpoint.resume_from_checkpoint == str(resume.resolve())
    assert model.model_kwargs == {'cli_runtime_test': 'yes'}


def test_dev_route_switch_is_opt_in(monkeypatch):
    monkeypatch.delenv('USE_SWIFT_V5', raising=False)
    assert resolve_route('sft', ROUTE_MAPPING) == 'swift.cli.sft'
    assert resolve_route('export', ROUTE_MAPPING) == 'swift.cli.export'

    monkeypatch.setenv('USE_SWIFT_V5', '1')
    assert resolve_route('sft', ROUTE_MAPPING) == 'swift.dev.cli.sft'
    assert resolve_route('export', ROUTE_MAPPING) == 'swift.dev.cli.export'
    megatron_routes = {'sft': 'swift.cli._megatron.sft'}
    assert resolve_route('sft', megatron_routes, is_megatron=True) == 'swift.dev.cli.megatron'


def test_dev_torchrun_route_executes_as_module(monkeypatch):
    captured = {}
    monkeypatch.setenv('USE_SWIFT_V5', '1')
    monkeypatch.setattr('sys.argv', ['swift-megatron', 'sft', '--model', 'm'])
    monkeypatch.setattr('swift.cli.main.get_torchrun_args', lambda: ['--nproc_per_node', '2'])

    def fake_run(args):
        captured['args'] = args
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr('swift.cli.main.subprocess.run', fake_run)
    cli_main({'sft': 'swift.cli._megatron.sft'}, is_megatron=True)
    assert captured['args'][-4:] == ['--module', 'swift.dev.cli.megatron', '--model', 'm']


def test_export_parses_actions_into_existing_configs():
    configs = parse_export_configs([
        '--model', 'm', '--adapters', 'a', '--merge_lora', 'true', '--quant_method', 'gptq',
        '--quant_bits', '4', '--quant_n_samples', '12', '--quant_batch_size', '2', '--group_size', '64',
        '--dataset', 'd', '--output_dir', 'out', '--exist_ok', 'true', '--test_convert_dtype', 'bfloat16',
    ])
    assert configs['convert_config'].merge_lora is True
    assert configs['convert_config'].exist_ok is True
    assert configs['quantize_config'].quant_method == 'gptq'
    assert configs['quantize_config'].quant_bits == 4
    assert configs['quantize_config'].quant_n_samples == 12
    assert configs['tuner_config'].adapters == ['a']
    assert configs['checkpoint_config'].output_dir == 'out'


def test_all_v5_routes_are_complete(monkeypatch):
    from swift.cli.main import DEV_MEGATRON_ROUTE_MAPPING, DEV_ROUTE_MAPPING

    monkeypatch.setenv('USE_SWIFT_V5', '1')
    for command in ('pt', 'sft', 'rlhf', 'infer', 'merge-lora', 'deploy', 'rollout', 'sample', 'export', 'eval', 'app'):
        assert resolve_route(command, ROUTE_MAPPING) == DEV_ROUTE_MAPPING[command]
    assert resolve_route('web-ui', ROUTE_MAPPING) == ROUTE_MAPPING['web-ui']
    legacy_megatron = {command: f'swift.cli._megatron.{command}' for command in ('pt', 'sft', 'rlhf', 'export')}
    for command, module in DEV_MEGATRON_ROUTE_MAPPING.items():
        assert resolve_route(command, legacy_megatron, is_megatron=True) == module


def test_all_cli_parsers_accept_minimal_argv(tmp_path):
    from swift.dev.cli.app import parse_app_configs
    from swift.dev.cli.deploy import parse_deploy_configs
    from swift.dev.cli.eval import parse_eval_configs
    from swift.dev.cli.infer import parse_infer_configs
    from swift.dev.cli.merge_lora import parse_merge_lora_configs
    from swift.dev.cli.rlhf import parse_rlhf_configs
    from swift.dev.cli.rollout import parse_rollout_configs
    from swift.dev.cli.sample import parse_sample_configs

    assert parse_infer_configs(['--model', 'm'])['model_config'].model == 'm'
    assert parse_deploy_configs(['--model', 'm'])['model_config'].model == 'm'
    assert parse_rollout_configs(['--model', 'm'])['rollout_config'].vllm_mode == 'server'
    assert parse_sample_configs(['--model', 'm', '--dataset', 'd'])['sampling_config'].sampler_engine == 'transformers'
    assert parse_eval_configs(['--model', 'm', '--eval_url', 'http://localhost:8000'])['eval_config'].eval_url
    assert parse_app_configs(['--model', 'm', '--base_url', 'http://localhost:8000'])['app_config'].base_url
    assert parse_merge_lora_configs(['--model', 'm', '--adapters', 'a'])['tuner_config'].adapters == ['a']
    for rlhf_type in ('dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm', 'grpo', 'ppo', 'gkd'):
        configs = parse_rlhf_configs(['--model', 'm', '--dataset', 'd', '--rlhf_type', rlhf_type])
        assert configs['rlhf_config'].rlhf_type == rlhf_type


def test_megatron_rlhf_parser_applies_megatron_surface(monkeypatch):
    from swift.dev.cli.rlhf import parse_rlhf_configs

    monkeypatch.setenv('WORLD_SIZE', '4')
    configs = parse_rlhf_configs([
        '--model', 'm', '--dataset', 'd', '--rlhf_type', 'dpo', '--lr', '0.0002', '--micro_batch_size', '2',
        '--global_batch_size', '8', '--bf16', 'true', '--attention_backend', 'fused',
    ], megatron=True)
    assert configs['distributed_config'].backend == 'megatron'
    assert configs['distributed_config'].nproc_per_node == 4
    assert configs['train_config'].gradient_accumulation_steps == 1
    assert configs['model_config'].torch_dtype == 'bfloat16'
    assert configs['model_config'].attn_impl == 'fused'

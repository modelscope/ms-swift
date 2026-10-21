"""Fast tests for dev CLI self-parsing (argv -> atomic Configs)."""
from __future__ import annotations
import dataclasses

import pytest

from swift.cli.main import ROUTE_MAPPING, cli_main, resolve_route
from swift.dev.cli.export import parse_export_configs
from swift.dev.cli.sft import parse_sft_configs
from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    QuantizeConfig,
    RLConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
    bootstrap_run,
    process_configs,
)
from swift.dev.recipe.utils import _resolve_step_interval


def test_defaults_come_from_configs():
    configs = parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--tuner_type', 'lora'])
    template, train, checkpoint = configs['template_config'], configs['train_config'], configs['checkpoint_config']
    logging, tuner, quantize = configs['logging_config'], configs['tuner_config'], configs['quantize_config']
    assert template.padding_side == TemplateConfig().padding_side == 'right'
    assert train.warmup_ratio == TrainConfig().warmup_ratio == 0.0
    assert train.optim == TrainConfig().optim == 'adamw_torch_fused'
    assert checkpoint.output_dir == CheckpointConfig().output_dir == 'output'
    assert logging.logging_steps == LoggingConfig().logging_steps == 5
    assert tuner.lora_rank == TunerConfig().lora_rank
    assert quantize == QuantizeConfig()


def test_explicit_values_land_in_owning_configs():
    configs = parse_sft_configs([
        '--model', 'm', '--load_args', 'false', '--dataset', 'd1', 'd2', '--val_dataset', 'v', '--torch_dtype', 'bfloat16',
        '--padding_side', 'left', '--max_length', '256', '--learning_rate', '0.0001', '--save_steps', '500',
        '--eval_steps', '100', '--logging_steps', '2', '--cp_comm_type', 'p2p', '--tuner_type', 'lora', '--lora_rank',
        '16',
        '--lora_alpha', '64', '--target_modules', 'q_proj',
    ])
    model, template, dataset = configs['model_config'], configs['template_config'], configs['dataset_config']
    train, dist, checkpoint = configs['train_config'], configs['distributed_config'], configs['checkpoint_config']
    logging, tuner, quantize = configs['logging_config'], configs['tuner_config'], configs['quantize_config']
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

    configs = parse_sft_configs([
        '--model', 'm', '--load_args', 'false', '--dataset', 'd', '--model_kwargs', '{"x": 1}', '--liger_kernel_config',
        '{"fused_linear_cross_entropy": true}', '--fsdp_config', '{"limit_all_gathers": true}',
    ])
    model, train, dist = configs['model_config'], configs['train_config'], configs['distributed_config']
    process_configs(model, TemplateConfig(), DatasetConfig(), train, dist)
    assert model.model_kwargs == {'x': 1}
    assert train.liger_kernel_config == {'fused_linear_cross_entropy': True}
    assert dist.fsdp_config == {'limit_all_gathers': True}


def test_legacy_precision_aliases_map_to_torch_dtype():
    model = parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--bf16', 'true'])['model_config']
    assert model.torch_dtype == 'bfloat16'
    with pytest.raises(ValueError, match='Conflicting values for --torch_dtype'):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--bf16', 'true', '--fp16', 'true'])


def test_full_training_yields_no_tuner_config():
    configs = parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--tuner_type', 'full'])
    assert configs['tuner_config'] is None
    assert configs['quantize_config'].quant_method is None


def test_fractional_intervals_survive_parse_for_step_planning():
    configs = parse_sft_configs(
        ['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--save_steps', '0.25', '--eval_steps', '0.1'])
    assert configs['checkpoint_config'].save_steps == 0.25
    assert configs['train_config'].eval_steps == 0.1


def test_unknown_and_unwired_flags_fail_loudly():
    with pytest.raises(ValueError, match='Unrecognized'):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--definitely_unknown', 'x'])
    # ``--optimizer`` is the Megatron-side optimizer (Literal adam/sgd/muon/dist_muon) while ``--optim``
    # names the transformers one; an out-of-Literal value dies at argparse rather than folding silently.
    with pytest.raises(SystemExit):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--optimizer', 'not_an_optimizer'])
    configs = parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--optimizer', 'muon'])
    assert configs['train_config'].optimizer == 'muon'
    assert configs['train_config'].optim == 'adamw_torch_fused'  # the transformers knob is untouched
    # ``--tuner`` is a Literal (full/lora/trainable_tokens): an unsupported tuner dies at argparse.
    with pytest.raises(SystemExit):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--tuner_type', 'vera'])


def test_megatron_aliases_work_on_transformers_backend():
    train = parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--lr', '0.001'])['train_config']
    assert train.learning_rate == 0.001


@pytest.mark.parametrize('argv', [
    ['--lr', '0.001', '--learning_rate=0.001'],
    ['--lr=1', '--learning-rate', '1.0'],
])
def test_aliases_accept_space_equals_hyphens_and_equal_double_writes(argv):
    train = parse_sft_configs(['--model=m', '--dataset', 'd', *argv])['train_config']
    assert train.learning_rate == pytest.approx(float(argv[1] if argv[0] == '--lr' else argv[0].split('=', 1)[1]))
    assert 'learning_rate' in train._explicit_fields


def test_aliases_reject_conflicting_double_writes():
    with pytest.raises(ValueError, match='Conflicting values for --learning_rate'):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--lr=0.001', '--learning-rate', '0.002'])
    with pytest.raises(ValueError, match='Conflicting values for --torch_dtype'):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--bf16=1', '--torch-dtype', 'float16'])


def test_false_precision_alias_does_not_override_canonical_dtype():
    model = parse_sft_configs(
        ['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--bf16=0', '--torch_dtype', 'float16'])['model_config']
    assert model.torch_dtype == 'float16'


def test_unconsumed_existing_configs_fail_loudly():
    logging = parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--report_to', 'wandb'])['logging_config']
    assert logging.report_to == ['wandb']
    quantize = parse_sft_configs(
        ['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--tuner_type', 'lora', '--quant_method', 'bnb', '--quant_bits',
         '4'])['quantize_config']
    assert quantize.quant_method == 'bnb' and quantize.quant_bits == 4
    with pytest.raises(ValueError, match='unsupported by `sft`'):
        parse_sft_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--top_p', '0.9'])


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


def test_multi_turn_bounds_are_validated():
    """max_turns lives on RolloutConfig now: the shared bounds check rejects 0 turns and a
    non-positive trajectory cap, and passes a sane combination."""
    from swift.dev.config import RolloutConfig
    from swift.dev.config.validate import validate_multi_turn_config

    with pytest.raises(ValueError, match='max_turns must be >= 1'):
        validate_multi_turn_config(RolloutConfig(max_turns=0))
    with pytest.raises(ValueError, match='max_trajectory_tokens must be >= 1'):
        validate_multi_turn_config(RolloutConfig(max_turns=4, max_trajectory_tokens=0))
    validate_multi_turn_config(RolloutConfig(max_turns=4, max_trajectory_tokens=8192))
    validate_multi_turn_config(RolloutConfig())  # max_turns=None -> single-turn, no constraint


def test_bootstrap_versions_output_and_abspaths_resume(tmp_path):
    resume = tmp_path / 'checkpoint-1'
    resume.mkdir()
    checkpoint = CheckpointConfig(output_dir=str(tmp_path / 'run'), resume_from_checkpoint=str(resume))
    model = ModelConfig(model_kwargs={'cli_runtime_test': 'yes'})
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
    # twinkle unifies the Megatron/Transformers stacks, so v5 has no separate `megatron` command: the
    # backend is a single flag, and routing the legacy `megatron` command redirects instead of resolving.
    with pytest.raises(RuntimeError, match='--backend megatron'):
        resolve_route('sft', {'sft': 'swift.cli._megatron.sft'}, is_megatron=True)


def test_dev_torchrun_route_executes_as_module(monkeypatch):
    captured = {}
    monkeypatch.setenv('USE_SWIFT_V5', '1')
    monkeypatch.setattr('sys.argv', ['swift', 'export', '--model', 'm', '--backend', 'megatron'])
    monkeypatch.setattr('swift.cli.main.get_torchrun_args', lambda: ['--nproc_per_node', '2'])

    class FakeProcess:
        returncode = 0

        def wait(self):
            return 0

    def fake_popen(args):
        captured['args'] = args
        return FakeProcess()

    monkeypatch.setattr('swift.cli.main.subprocess.Popen', fake_popen)
    cli_main()
    # `swift export --backend megatron` is the single distributed Megatron export entry, so the dev
    # export route relaunches under torchrun as a module when NPROC_PER_NODE/NNODES is set.
    assert captured['args'][-6:] == [
        '--module', 'swift.dev.cli.export', '--model', 'm', '--backend', 'megatron'
    ]


def test_export_parses_actions_into_existing_configs():
    configs = parse_export_configs([
        '--model', 'm', '--load_args', 'false', '--adapters', 'a', '--merge_lora', 'true', '--quant_method', 'gptq',
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
    from swift.cli.main import DEV_ROUTE_MAPPING

    monkeypatch.setenv('USE_SWIFT_V5', '1')
    # Every dev route resolves to its dev module; the legacy `merge-lora` command is `merge` in v5.
    for command, module in DEV_ROUTE_MAPPING.items():
        assert resolve_route(command, ROUTE_MAPPING) == module
    assert resolve_route('web-ui', ROUTE_MAPPING) == ROUTE_MAPPING['web-ui']
    # v5 has no separate `megatron` command for any of the unified commands: routing one redirects.
    for command in ('pt', 'sft', 'rlhf', 'export'):
        with pytest.raises(RuntimeError, match='--backend megatron'):
            resolve_route(command, ROUTE_MAPPING, is_megatron=True)


def test_all_cli_parsers_accept_minimal_argv(tmp_path):
    from swift.dev.cli.deploy import parse_deploy_configs
    from swift.dev.cli.eval import parse_eval_configs
    from swift.dev.cli.infer import parse_infer_configs
    from swift.dev.cli.merge import parse_merge_configs
    from swift.dev.cli.rl import parse_rl_configs

    assert parse_infer_configs(['--model', 'm', '--load_args', 'false'])['model_config'].model == 'm'
    assert parse_deploy_configs(['--model', 'm', '--load_args', 'false'])['model_config'].model == 'm'
    assert parse_eval_configs(['--model', 'm', '--load_args', 'false', '--eval_dataset', 'bench'])['eval_config'].eval_dataset == ['bench']
    assert parse_merge_configs(['--model', 'm', '--load_args', 'false', '--adapters', 'a'])['tuner_config'].adapters == ['a']
    for rl_type in ('dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm', 'grpo', 'ppo', 'gkd'):
        configs = parse_rl_configs(['--model', 'm', '--load_args', 'false', '--dataset', 'd', '--rlhf_type', rl_type])
        assert configs['rl_config'].rl_type == rl_type


def test_megatron_rlhf_parser_applies_megatron_surface(monkeypatch):
    from swift.dev.cli.rl import parse_rl_configs

    monkeypatch.setenv('WORLD_SIZE', '4')
    configs = parse_rl_configs([
        '--model', 'm', '--load_args', 'false', '--dataset', 'd', '--rlhf_type', 'dpo', '--lr', '0.0002', '--micro_batch_size', '2',
        '--global_batch_size', '8', '--bf16', 'true', '--attention_backend', 'fused', '--backend', 'megatron',
    ])
    assert configs['distributed_config'].backend == 'megatron'
    assert configs['distributed_config'].nproc_per_node == 4
    assert configs['train_config'].gradient_accumulation_steps == 1
    assert configs['model_config'].torch_dtype == 'bfloat16'
    assert configs['model_config'].attn_impl == 'fused'

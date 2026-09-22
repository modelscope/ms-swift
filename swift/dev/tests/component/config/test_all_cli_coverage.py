"""Mechanical legacy-field coverage for every non-SFT v5 CLI."""
import dataclasses

import pytest

from swift.arguments import (
    AppArguments,
    DeployArguments,
    EvalArguments,
    ExportArguments,
    InferArguments,
    PretrainArguments,
    RLHFArguments,
    RolloutArguments,
    SamplingArguments,
    SftArguments,
)
from swift.dev.cli.app import parse_app_configs
from swift.dev.cli.deploy import DeployCliConfig, parse_deploy_configs
from swift.dev.cli.eval import parse_eval_configs
from swift.dev.cli.export import parse_export_configs
from swift.dev.cli.infer import InferCliConfig, parse_infer_configs
from swift.dev.cli.legacy_coverage import (
    COMMAND_RUNTIME_CONSUMERS,
    audit_config_consumers,
    build_legacy_contract,
    classified_fields,
    unsupported_contracts,
)
from swift.dev.cli.rlhf import RlhfCliCompatConfig, parse_rlhf_configs
from swift.dev.cli.rollout import parse_rollout_configs
from swift.dev.cli.sample import parse_sample_configs
from swift.dev.cli.sft import SftCliCompatConfig
from swift.dev.config import (
    AppConfig,
    CheckpointConfig,
    ConvertConfig,
    DatasetConfig,
    DeployConfig,
    DistributedConfig,
    EvalConfig,
    GenerationConfig,
    InferConfig,
    LoggingConfig,
    MegatronConfig,
    ModelConfig,
    MoEConfig,
    QuantizeConfig,
    RLHFConfig,
    RolloutConfig,
    RuntimeConfig,
    SamplingConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)

_BASE = [ModelConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig]
_TRAIN = _BASE + [TrainConfig, DistributedConfig, LoggingConfig, QuantizeConfig, SftCliCompatConfig]
CLI_SURFACES = {
    'pt': (PretrainArguments, _TRAIN),
    'sft': (SftArguments, _TRAIN),
    'rlhf': (RLHFArguments, _TRAIN + [GenerationConfig, RolloutConfig, RLHFConfig, MegatronConfig, MoEConfig,
                                     RlhfCliCompatConfig]),
    'infer': (InferArguments, _BASE + [DistributedConfig, GenerationConfig, RolloutConfig, InferConfig,
                                      QuantizeConfig, InferCliConfig, RuntimeConfig]),
    'deploy': (DeployArguments, _BASE + [GenerationConfig, RolloutConfig, InferConfig, DeployConfig,
                                         QuantizeConfig, DeployCliConfig, RuntimeConfig]),
    'rollout': (RolloutArguments,
                _BASE + [GenerationConfig, RolloutConfig, RLHFConfig, DeployConfig, QuantizeConfig, RuntimeConfig]),
    'sample': (SamplingArguments, _BASE + [DistributedConfig, GenerationConfig, RolloutConfig, InferConfig,
                                           SamplingConfig, RLHFConfig, QuantizeConfig, RuntimeConfig]),
    'eval': (EvalArguments, _BASE + [GenerationConfig, RolloutConfig, InferConfig, DeployConfig, EvalConfig,
                                     QuantizeConfig, DeployCliConfig, RuntimeConfig]),
    'app': (AppArguments, _BASE + [GenerationConfig, RolloutConfig, InferConfig, DeployConfig, AppConfig,
                                   QuantizeConfig, DeployCliConfig, RuntimeConfig]),
    'export': (ExportArguments, _BASE + [DistributedConfig, QuantizeConfig, ConvertConfig, GenerationConfig,
                                         RuntimeConfig]),
}


@pytest.mark.parametrize('command', CLI_SURFACES)
def test_legacy_only_fields_are_exhaustively_classified(command):
    legacy_class, config_classes = CLI_SURFACES[command]
    legacy_fields = {field.name for field in dataclasses.fields(legacy_class)}
    config_fields = {field.name for cls in config_classes for field in dataclasses.fields(cls)}
    classified = classified_fields(command).intersection(legacy_fields)
    assert legacy_fields - config_fields <= classified


@pytest.mark.parametrize('command', CLI_SURFACES)
def test_legacy_contract_partitions_every_field_once(command):
    legacy_class, config_classes = CLI_SURFACES[command]
    legacy_fields = {field.name for field in dataclasses.fields(legacy_class)}
    contract, unaccounted = build_legacy_contract(command, legacy_fields, config_classes)
    assert not unaccounted
    assert set(contract) == legacy_fields
    assert all(item.kind in {'direct', 'alias', 'derived', 'unsupported'} for item in contract.values())
    assert all(item.consumer for item in contract.values() if item.kind != 'unsupported')


@pytest.mark.parametrize('command', CLI_SURFACES)
def test_command_runtime_consumer_is_a_real_entrypoint(command):
    import importlib

    module_name, function_name = COMMAND_RUNTIME_CONSUMERS[command].rsplit('.', 1)
    assert callable(getattr(importlib.import_module(module_name), function_name))


@pytest.mark.parametrize('command', CLI_SURFACES)
def test_every_accepted_config_field_has_an_owner_consumer(command):
    _, config_classes = CLI_SURFACES[command]
    consumers = audit_config_consumers(command, config_classes)
    expected = {field.name for cls in config_classes for field in dataclasses.fields(cls)}
    assert set(consumers) == expected
    assert all('swift.dev.cli.' in consumer and ' -> ' in consumer for consumer in consumers.values())


@pytest.mark.parametrize('command', CLI_SURFACES)
def test_unsupported_contracts_explain_reason_and_alternative(command):
    for item in unsupported_contracts(command).values():
        assert item.reason
        assert item.replacement


@pytest.mark.parametrize(
    ('parser', 'argv', 'error'),
    [
        (parse_infer_configs, ['--model', 'm', '--lmdeploy_tp', '2'], ValueError),
        (parse_deploy_configs, ['--model', 'm', '--use_ray', 'true'], ValueError),
        (parse_rollout_configs, ['--model', 'm', '--result_path', 'x'], ValueError),
        (parse_eval_configs, ['--model', 'm', '--use_swift_lora', 'true'], ValueError),
        (parse_app_configs, ['--model', 'm', '--ignore_args_error', 'true'], ValueError),
        (parse_export_configs, ['--model', 'm', '--use_swift_lora', 'true'], ValueError),
    ],
)
def test_classified_legacy_fields_fail_with_explicit_reason(parser, argv, error):
    with pytest.raises(error):
        parser(argv)


def test_rlhf_response_length_alias_and_deprecated_seq_kd():
    configs = parse_rlhf_configs(['--model', 'm', '--dataset', 'd', '--response_length', '128'])
    assert configs['rlhf_config'].max_completion_length == 128
    with pytest.raises(ValueError, match='deprecated'):
        parse_rlhf_configs(['--model', 'm', '--dataset', 'd', '--seq_kd', 'true'])


def test_runtime_seed_and_sampling_legacy_defaults():
    infer = parse_infer_configs(['--model', 'm', '--seed', '7'])
    assert infer['runtime_config'].seed == 7
    sample = parse_sample_configs(['--model', 'm', '--dataset', 'd'])
    assert sample['sampling_config'].num_return_sequences == 64
    assert sample['sampling_config'].n_best_to_keep == 5
    assert sample['sampling_config'].batch_size == 1
    assert sample['sampling_config'].output_file.endswith('.jsonl')
    assert sample['template_config'].padding_side == 'left'


def test_eval_and_app_accept_merge_lora_for_local_deploy_only():
    eval_configs = parse_eval_configs(['--model', 'm', '--merge_lora', 'true'])
    app_configs = parse_app_configs(['--model', 'm', '--merge_lora', 'true'])
    assert eval_configs['cli_config'].merge_lora is True
    assert app_configs['cli_config'].merge_lora is True

    with pytest.raises(ValueError, match='local deployment'):
        parse_eval_configs(['--model', 'm', '--eval_url', 'http://localhost:8000', '--merge_lora', 'true'])
    with pytest.raises(ValueError, match='local deployment'):
        parse_app_configs(['--model', 'm', '--base_url', 'http://localhost:8000', '--merge_lora', 'true'])

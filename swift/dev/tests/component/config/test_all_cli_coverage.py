"""Mechanical legacy-field coverage for every non-SFT v5 CLI."""
import dataclasses

import pytest

from swift.arguments import (
    AppArguments,
    DeployArguments,
    EvalArguments,
    ExportArguments,
    InferArguments,
    RLHFArguments,
    RolloutArguments,
    SamplingArguments,
)
from swift.dev.cli.app import parse_app_configs
from swift.dev.cli.deploy import DeployCliConfig, parse_deploy_configs
from swift.dev.cli.eval import parse_eval_configs
from swift.dev.cli.export import parse_export_configs
from swift.dev.cli.infer import InferCliConfig, parse_infer_configs
from swift.dev.cli.legacy_coverage import classified_fields
from swift.dev.cli.rlhf import _RLHF_EXTENSION_FIELDS, RlhfCliCompatConfig, parse_rlhf_configs
from swift.dev.cli.rollout import parse_rollout_configs
from swift.dev.cli.sample import parse_sample_configs
from swift.dev.cli.sft import LEGACY_ONLY_CLASSIFICATION, SftCliCompatConfig
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
    ModelConfig,
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
CLI_SURFACES = {
    'infer': (InferArguments, _BASE + [DistributedConfig, GenerationConfig, RolloutConfig, InferConfig,
                                      InferCliConfig, RuntimeConfig]),
    'deploy': (DeployArguments, _BASE + [GenerationConfig, RolloutConfig, InferConfig, DeployConfig,
                                         DeployCliConfig, RuntimeConfig]),
    'rollout': (RolloutArguments, _BASE + [GenerationConfig, RolloutConfig, RLHFConfig, DeployConfig, RuntimeConfig]),
    'sample': (SamplingArguments, _BASE + [DistributedConfig, GenerationConfig, RolloutConfig, InferConfig,
                                           SamplingConfig, RLHFConfig, RuntimeConfig]),
    'eval': (EvalArguments, _BASE + [GenerationConfig, RolloutConfig, InferConfig, DeployConfig, EvalConfig,
                                     DeployCliConfig, RuntimeConfig]),
    'app': (AppArguments, _BASE + [GenerationConfig, RolloutConfig, InferConfig, DeployConfig, AppConfig,
                                   DeployCliConfig, RuntimeConfig]),
    'export': (ExportArguments, _BASE + [DistributedConfig, QuantizeConfig, ConvertConfig, GenerationConfig,
                                         RuntimeConfig]),
}


@pytest.mark.parametrize('command', CLI_SURFACES)
def test_legacy_only_fields_are_exhaustively_classified(command):
    legacy_class, config_classes = CLI_SURFACES[command]
    legacy_fields = {field.name for field in dataclasses.fields(legacy_class)}
    config_fields = {field.name for cls in config_classes for field in dataclasses.fields(cls)}
    assert legacy_fields - config_fields == classified_fields(command)


@pytest.mark.parametrize(
    ('parser', 'argv', 'error'),
    [
        (parse_infer_configs, ['--model', 'm', '--lmdeploy_tp', '2'], ValueError),
        (parse_deploy_configs, ['--model', 'm', '--use_ray', 'true'], ValueError),
        (parse_rollout_configs, ['--model', 'm', '--result_path', 'x'], ValueError),
        (parse_sample_configs, ['--model', 'm', '--quant_method', 'bnb'], NotImplementedError),
        (parse_eval_configs, ['--model', 'm', '--use_swift_lora', 'true'], ValueError),
        (parse_app_configs, ['--model', 'm', '--ignore_args_error', 'true'], ValueError),
        (parse_export_configs, ['--model', 'm', '--use_swift_lora', 'true'], ValueError),
    ],
)
def test_classified_legacy_fields_fail_with_explicit_reason(parser, argv, error):
    with pytest.raises(error):
        parser(argv)


def test_rlhf_legacy_only_fields_are_exhaustively_classified():
    config_classes = _BASE + [TrainConfig, DistributedConfig, LoggingConfig, GenerationConfig, RolloutConfig,
                              RLHFConfig, SftCliCompatConfig, RlhfCliCompatConfig]
    legacy_fields = {field.name for field in dataclasses.fields(RLHFArguments)}
    config_fields = {field.name for cls in config_classes for field in dataclasses.fields(cls)}
    classified = {name for names in LEGACY_ONLY_CLASSIFICATION.values() for name in names}
    classified.update(_RLHF_EXTENSION_FIELDS)
    classified.update({
        'bnb_4bit_compute_dtype', 'bnb_4bit_quant_storage', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant',
        'hqq_axis', 'quant_bits', 'quant_method'
    })
    assert legacy_fields - config_fields == classified


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

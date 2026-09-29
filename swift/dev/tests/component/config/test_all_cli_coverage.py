"""Mechanical legacy-field coverage for every non-SFT v5 CLI.

Each entry pairs a legacy v4 ``*Arguments`` class with the exact Config classes the matching
``parse_*_configs`` builds, so ``build_legacy_contract`` sees the same owners the real parser does.
The surface mirrors the commands that still exist (app/rollout/sample were folded into infer or
removed), and only the surviving compatibility mechanisms are exercised: the consumer-audit registry
was deleted, so there is no consumer assertion here.
"""
import dataclasses

import pytest

from swift.arguments import (
    DeployArguments,
    EvalArguments,
    ExportArguments,
    InferArguments,
    PretrainArguments,
    RLHFArguments,
    SftArguments,
)
from swift.dev.cli.deploy import parse_deploy_configs
from swift.dev.cli.eval import parse_eval_configs
from swift.dev.cli.export import parse_export_configs
from swift.dev.cli.infer import InferCliConfig, parse_infer_configs
from swift.dev.cli.legacy_coverage import (
    LEGACY_ALIASES,
    build_legacy_contract,
    classified_fields,
    unsupported_contracts,
)
from swift.dev.cli.merge import MergeCliConfig, parse_merge_configs
from swift.dev.cli.rlhf import RlhfCliCompatConfig, parse_rlhf_configs
from swift.dev.config import (
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
    PluginConfig,
    QuantizeConfig,
    RLHFConfig,
    RolloutConfig,
    RuntimeConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)

# The Transformers training surface shared by pt/sft; rlhf extends it with the generation/rollout knobs.
_TRAIN = [
    ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig,
    LoggingConfig, TunerConfig, QuantizeConfig, MegatronConfig, MoEConfig
]
CLI_SURFACES = {
    'pt': (PretrainArguments, _TRAIN),
    'sft': (SftArguments, _TRAIN),
    'rlhf': (RLHFArguments, _TRAIN + [GenerationConfig, RolloutConfig, RLHFConfig, RlhfCliCompatConfig]),
    'infer': (InferArguments, [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig, TunerConfig,
        GenerationConfig, RolloutConfig, InferConfig, RLHFConfig, QuantizeConfig, InferCliConfig, RuntimeConfig
    ]),
    'deploy': (DeployArguments, [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig, TunerConfig,
        GenerationConfig, RolloutConfig, InferConfig, DeployConfig, QuantizeConfig, RuntimeConfig
    ]),
    'eval': (EvalArguments, [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig, RolloutConfig,
        InferConfig, EvalConfig, QuantizeConfig, RuntimeConfig
    ]),
    'export': (ExportArguments, [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig, QuantizeConfig,
        ConvertConfig, TunerConfig, RuntimeConfig, MegatronConfig, MoEConfig
    ]),
    # legacy `swift merge-lora` parsed ExportArguments too, so merge's legacy surface is the same class;
    # merge keeps only the LoRA-folding subset and rejects the quant/convert/generation operations.
    'merge': (ExportArguments, [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig, MergeCliConfig,
        RuntimeConfig
    ]),
}


@pytest.mark.parametrize('command', CLI_SURFACES)
def test_legacy_only_fields_are_exhaustively_classified(command):
    legacy_class, config_classes = CLI_SURFACES[command]
    legacy_fields = {field.name for field in dataclasses.fields(legacy_class)}
    config_fields = {field.name for cls in config_classes for field in dataclasses.fields(cls)}
    # A legacy field is accounted for if a parsed Config owns it, LEGACY_ALIASES remaps it before parse,
    # or CLI_LEGACY_ONLY explicitly rejects it -- the same three categories build_legacy_contract uses.
    classified = classified_fields(command).intersection(legacy_fields)
    assert legacy_fields - config_fields - set(LEGACY_ALIASES) <= classified


@pytest.mark.parametrize('command', CLI_SURFACES)
def test_legacy_contract_partitions_every_field_once(command):
    legacy_class, config_classes = CLI_SURFACES[command]
    legacy_fields = {field.name for field in dataclasses.fields(legacy_class)}
    contract, unaccounted = build_legacy_contract(command, legacy_fields, config_classes)
    assert not unaccounted
    assert set(contract) == legacy_fields
    assert all(item.kind in {'direct', 'alias', 'unsupported'} for item in contract.values())


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
        (parse_eval_configs, ['--model', 'm', '--use_swift_lora', 'true'], ValueError),
        (parse_export_configs, ['--model', 'm', '--use_swift_lora', 'true'], ValueError),
        (parse_merge_configs, ['--model', 'm', '--quant_method', 'bnb'], ValueError),
        (parse_merge_configs, ['--model', 'm', '--exist_ok', 'true'], ValueError),
    ],
)
def test_classified_legacy_fields_fail_with_explicit_reason(parser, argv, error):
    with pytest.raises(error):
        parser(argv)


def test_rlhf_response_length_alias_and_removed_seq_kd():
    configs = parse_rlhf_configs(['--model', 'm', '--dataset', 'd', '--response_length', '128'])
    assert configs['rlhf_config'].max_completion_length == 128
    # seq_kd is a removed option, refused by reject_legacy_only_flags before parsing reaches the
    # (now unreachable) RlhfCliCompatConfig.seq_kd branch.
    with pytest.raises(ValueError, match='unsupported'):
        parse_rlhf_configs(['--model', 'm', '--dataset', 'd', '--seq_kd', 'true'])


def test_runtime_seed_lands_on_runtime_config():
    # --load_args false keeps the parse off the checkpoint-args restore path, which would otherwise try
    # to fetch the fake model id 'm' from the hub.
    infer = parse_infer_configs(['--model', 'm', '--seed', '7', '--load_args', 'false'])
    assert infer['runtime_config'].seed == 7

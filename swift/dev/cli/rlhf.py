"""RLHF CLI: atomic Config parsing and algorithm dispatch."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RlhfCliCompatConfig:
    """Deprecated RLHF aliases retained for command-line compatibility."""

    response_length: Optional[int] = None
    seq_kd: bool = False


# Kept as a compatibility export for the coverage test; all former extension fields now live on RLHFConfig.
_RLHF_EXTENSION_FIELDS = ()


def _configure_megatron(model_config, train_config, distributed_config, tuner_config):
    from swift.dev.cli.megatron import _derive_megatron_ga, _fix_mtp, _select_megatron_tuner

    distributed_config.backend = 'megatron'
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed_config.nproc_per_node = distributed_config.nproc_per_node or world_size
    _fix_mtp(model_config)
    if train_config.weight_decay_incr_style == 'constant':
        train_config.start_weight_decay = None
        train_config.end_weight_decay = None
    _derive_megatron_ga(train_config, distributed_config, world_size)
    return _select_megatron_tuner(tuner_config)


def _apply_rlhf_compat(passed, generation_config, rlhf_config, rlhf_compat) -> None:
    if 'max_new_tokens' in passed and generation_config.max_new_tokens is not None:
        rlhf_config.max_completion_length = generation_config.max_new_tokens
    if rlhf_compat.response_length is not None:
        rlhf_config.max_completion_length = rlhf_compat.response_length
    if rlhf_compat.seq_kd:
        raise ValueError('`--seq_kd` is deprecated and was never implemented; use the GKD objective directly.')


def parse_rlhf_configs(argv: Optional[List[str]] = None, *, megatron: bool = False) -> Dict[str, Any]:
    from swift.dev.cli.parser import flag_names, parse_configs_strict, resolve_argv, select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        PluginConfig,
        QuantizeConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    reject_legacy_only_flags('megatron_rlhf' if megatron else 'rlhf', effective_argv)
    passed = flag_names(effective_argv)
    classes = [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig,
        LoggingConfig, TunerConfig, GenerationConfig, RolloutConfig, RLHFConfig, QuantizeConfig, MegatronConfig,
        MoEConfig
    ]
    if megatron:
        from swift.dev.cli.megatron import MegatronCliCompatConfig
        classes.append(MegatronCliCompatConfig)
    classes.append(RlhfCliCompatConfig)
    owners = {
        'loss_scale': TemplateConfig,
        'loss_type': RLHFConfig,
        'max_new_tokens': GenerationConfig,
        'temperature': RLHFConfig,
        'reward_funcs': RLHFConfig,
        'reward_weights': RLHFConfig,
    }
    configs = parse_configs_strict(classes, effective_argv, command='swift rlhf', field_owners=owners)
    common = configs[:15]
    (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
     checkpoint_config, logging_config, tuner_config, generation_config, rollout_config, rlhf_config, quantize_config,
     megatron_config, moe_config) = common
    rlhf_compat = configs[-1]
    if megatron:
        tuner_config = _configure_megatron(model_config, train_config, distributed_config, tuner_config)
    else:
        tuner_config = select_tuner(tuner_config)

    _apply_rlhf_compat(passed, generation_config, rlhf_config, rlhf_compat)
    if rlhf_config.rlhf_type in {'grpo', 'ppo'}:
        rollout_config.use_vllm = True
        rollout_config.vllm_mode = rollout_config.vllm_mode or 'colocate'
        distributed_config.mode = 'ray'

    return {
        'model_config': model_config,
        'plugin_config': plugin_config,
        'template_config': template_config,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'checkpoint_config': checkpoint_config,
        'logging_config': logging_config,
        'tuner_config': tuner_config,
        'generation_config': generation_config,
        'rollout_config': rollout_config,
        'rlhf_config': rlhf_config,
        'quantize_config': quantize_config,
        'megatron_config': megatron_config,
        'moe_config': moe_config,
    }


def run_rlhf_configs(configs: Dict[str, Any]) -> List[dict]:
    """Process, validate, bootstrap, and execute an already parsed RLHF Config set."""
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_rlhf

    process_and_validate_configs(configs)
    return run_rlhf(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        configs['train_config'],
        configs['distributed_config'],
        configs['checkpoint_config'],
        configs['rollout_config'],
        configs['rlhf_config'],
        configs['tuner_config'],
        configs['generation_config'],
        configs['logging_config'],
        configs['quantize_config'],
        megatron_config=configs['megatron_config'],
        moe_config=configs['moe_config'],
        output_dir=configs['checkpoint_config'].output_dir,
    )


def rlhf_main(argv: Optional[List[str]] = None) -> List[dict]:
    return run_rlhf_configs(parse_rlhf_configs(argv))


if __name__ == '__main__':
    rlhf_main()

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


def _configure_megatron(model_config, train_config, distributed_config, tuner_config, compat):
    from swift.dev.cli.megatron import (
        _apply_megatron_precision,
        _attn_backend_name,
        _derive_megatron_ga,
        _fix_mtp,
        _reject_unmappable,
        _select_megatron_tuner,
    )

    _reject_unmappable(train_config)
    distributed_config.backend = 'megatron'
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed_config.nproc_per_node = distributed_config.nproc_per_node or world_size
    if compat.attention_backend is not None:
        model_config.attn_impl = _attn_backend_name(compat.attention_backend)
    _apply_megatron_precision(model_config, compat)
    _fix_mtp(model_config, model_config)
    if train_config.weight_decay_incr_style == 'constant':
        train_config.start_weight_decay = None
        train_config.end_weight_decay = None
    _derive_megatron_ga(train_config, distributed_config, world_size)
    return _select_megatron_tuner(tuner_config)


def _apply_rlhf_compat(passed, generation_config, rollout_config, rlhf_config, rlhf_compat) -> None:
    for name in ('top_k', 'top_p', 'repetition_penalty', 'stop_words', 'structured_outputs_regex'):
        if name in passed:
            setattr(rollout_config, name, getattr(generation_config, name))
    if 'max_new_tokens' in passed and generation_config.max_new_tokens is not None:
        rlhf_config.max_completion_length = generation_config.max_new_tokens
    if rlhf_compat.response_length is not None:
        rlhf_config.max_completion_length = rlhf_compat.response_length
    if rlhf_compat.seq_kd:
        raise ValueError('`--seq_kd` is deprecated and was never implemented; use the GKD objective directly.')


def parse_rlhf_configs(argv: Optional[List[str]] = None, *, megatron: bool = False) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import LOAD_QUANTIZATION_FIELDS
    from swift.dev.cli.parser import flag_names, parse_configs_strict, resolve_argv
    from swift.dev.cli.sft import (
        LEGACY_ONLY_CLASSIFICATION,
        SftCliCompatConfig,
        _apply_precision_aliases,
        _select_tuner,
    )
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        LoggingConfig,
        ModelConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    passed = flag_names(effective_argv)
    unsupported = {
        name for names in LEGACY_ONLY_CLASSIFICATION.values() for name in names
    } | set(LOAD_QUANTIZATION_FIELDS) | set(_RLHF_EXTENSION_FIELDS)
    matched = sorted(passed.intersection(unsupported))
    if matched:
        raise NotImplementedError(f'{matched} have no consumer in the dev RLHF recipes and are refused explicitly.')
    compat_class = SftCliCompatConfig
    if megatron:
        from swift.dev.cli.megatron import MegatronCliCompatConfig
        compat_class = MegatronCliCompatConfig
    classes = [
        ModelConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig, LoggingConfig,
        TunerConfig, GenerationConfig, RolloutConfig, RLHFConfig, compat_class, RlhfCliCompatConfig
    ]
    owners = {
        'loss_scale': TemplateConfig,
        'loss_type': RLHFConfig,
        'max_new_tokens': GenerationConfig,
        'temperature': RLHFConfig,
        'top_k': GenerationConfig,
        'top_p': GenerationConfig,
        'repetition_penalty': GenerationConfig,
        'stop_words': GenerationConfig,
        'structured_outputs_regex': GenerationConfig,
        'reward_funcs': RLHFConfig,
        'reward_weights': RLHFConfig,
    }
    configs = parse_configs_strict(classes, effective_argv, command='swift rlhf', field_owners=owners)
    (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
     logging_config, tuner_config, generation_config, rollout_config, rlhf_config, compat, rlhf_compat) = configs
    if megatron:
        tuner_config = _configure_megatron(model_config, train_config, distributed_config, tuner_config, compat)
    else:
        _apply_precision_aliases(model_config, compat)
        tuner_config = _select_tuner(tuner_config)

    _apply_rlhf_compat(passed, generation_config, rollout_config, rlhf_config, rlhf_compat)
    if rlhf_config.rlhf_type in {'grpo', 'ppo'}:
        rollout_config.use_vllm = True
        rollout_config.vllm_mode = rollout_config.vllm_mode or 'colocate'
        distributed_config.use_ray = True
        distributed_config.mode = 'ray'
    elif megatron and distributed_config.use_ray:
        distributed_config.mode = 'ray'

    return {
        'model_config': model_config,
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
    }


def run_rlhf_configs(configs: Dict[str, Any]) -> List[dict]:
    """Process, validate, bootstrap, and execute an already parsed RLHF Config set."""
    from swift.dev.cli.runtime import bootstrap_run
    from swift.dev.config import process_configs, validate_configs
    from swift.dev.recipe import run_rlhf

    process_configs(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        configs['train_config'],
        configs['distributed_config'],
        configs['checkpoint_config'],
        configs['tuner_config'],
        rlhf_config=configs['rlhf_config'],
    )
    validate_configs(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        configs['train_config'],
        configs['distributed_config'],
        configs['checkpoint_config'],
        configs['tuner_config'],
        configs['rlhf_config'],
        configs['logging_config'],
    )
    bootstrap_run(
        configs['model_config'], configs['checkpoint_config'], configs['dataset_config'], configs['tuner_config'],
        seed=configs['train_config'].seed)
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
        output_dir=configs['checkpoint_config'].output_dir,
    )


def rlhf_main(argv: Optional[List[str]] = None) -> List[dict]:
    return run_rlhf_configs(parse_rlhf_configs(argv))


if __name__ == '__main__':
    rlhf_main()

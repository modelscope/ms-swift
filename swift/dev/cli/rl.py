# Copyright (c) ModelScope Contributors. All rights reserved.
"""RL CLI: atomic Config parsing and algorithm dispatch."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RlCliCompatConfig:
    """Deprecated RLHF aliases retained for command-line compatibility."""

    response_length: Optional[int] = None
    seq_kd: bool = False


# Kept as a compatibility export for the coverage test; all former extension fields now live on RLConfig.
_RL_EXTENSION_FIELDS = ()


def _apply_rl_compat(passed, generation_config, rl_config, rlhf_compat) -> None:
    if 'max_new_tokens' in passed and generation_config.max_new_tokens is not None:
        rl_config.max_completion_length = generation_config.max_new_tokens
    if rlhf_compat.response_length is not None:
        rl_config.max_completion_length = rlhf_compat.response_length
    if rlhf_compat.seq_kd:
        raise ValueError('`--seq_kd` is deprecated and was never implemented; use the GKD objective directly.')


def parse_rl_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli._megatron_compat import configure_megatron, is_megatron_argv
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
        RewardConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    megatron = is_megatron_argv(effective_argv)
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    reject_legacy_only_flags('megatron_rl' if megatron else 'rl', effective_argv)
    passed = flag_names(effective_argv)
    classes = [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig,
        LoggingConfig, TunerConfig, GenerationConfig, RolloutConfig, RLConfig, RewardConfig, QuantizeConfig,
        MegatronConfig, MoEConfig
    ]
    classes.append(RlCliCompatConfig)
    # Every field name is unique across these Configs (the RL objective's distill temperature is
    # ``distill_temperature``, ``loss_scale`` lives only on TemplateConfig, ``max_new_tokens``/``temperature``
    # only on GenerationConfig), so no per-flag owner disambiguation is needed.
    configs = parse_configs_strict(classes, effective_argv, command='swift rl')
    common = configs[:16]
    (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
     checkpoint_config, logging_config, tuner_config, generation_config, rollout_config, rl_config, reward_config,
     quantize_config, megatron_config, moe_config) = common
    rlhf_compat = configs[-1]
    if megatron:
        tuner_config = configure_megatron(model_config, train_config, distributed_config, tuner_config)
    else:
        tuner_config = select_tuner(tuner_config)

    _apply_rl_compat(passed, generation_config, rl_config, rlhf_compat)
    # Every online algorithm -- grpo/ppo/rft and the distillation family gkd/opsd/mopd -- rolls out on a
    # separate vLLM/SGLang sampler over Ray and syncs the trained weights into it, so all of them force
    # the Ray + colocate-sampler placement. On-policy generation must be backend-general: a separate
    # sampler + weight sync works for both transformers and megatron, whereas in-process
    # ``model.generate`` is transformers-only.
    if rl_config.rl_type in {'grpo', 'ppo', 'rft', 'gkd', 'opsd', 'mopd'}:
        rollout_config.sampler_mode = rollout_config.sampler_mode or 'colocate'
        distributed_config.mode = 'ray'
    # The offline preference family (dpo/kto/cpo/orpo/simpo/rm) is RL too (RL_PLAN basic principle 2): it
    # runs mode='ray' with the policy as a Ray actor on the 'model' group, and a FULL-PARAMETER frozen
    # reference (dpo/kto full fine-tuning) as a separate actor on its own 'ref' group. It rolls out nothing,
    # so -- unlike the online family above -- it sets no vLLM sampler; only the mode is forced.
    elif rl_config.rl_type in {'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'}:
        distributed_config.mode = 'ray'
    else:
        # Fail loudly rather than silently leaving mode='local'. The two families above are hardcoded sets
        # kept in sync by hand with RLConfig.rl_type's Literal; a new rl_type added to the Literal but not
        # classified here would otherwise fall through and run a non-Ray path, which RL never supports.
        raise ValueError(
            f'--rl_type {rl_config.rl_type!r} is not classified in parse_rl_configs. Every rl_type must '
            f"force distributed_config.mode='ray' (RL has no mode='local' path): add it to the online family "
            f'(rolls out on a separate vLLM sampler) or the offline preference family (mode only) above.')

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
        'rl_config': rl_config,
        'reward_config': reward_config,
        'quantize_config': quantize_config,
        'megatron_config': megatron_config,
        'moe_config': moe_config,
    }


def run_rl_configs(configs: Dict[str, Any]) -> List[dict]:
    """Process, validate, bootstrap, and execute an already parsed RLHF Config set."""
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_rl

    process_and_validate_configs(configs)
    return run_rl(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        configs['train_config'],
        configs['distributed_config'],
        configs['checkpoint_config'],
        configs['rollout_config'],
        configs['rl_config'],
        configs['reward_config'],
        configs['tuner_config'],
        configs['generation_config'],
        configs['logging_config'],
        configs['quantize_config'],
        megatron_config=configs['megatron_config'],
        moe_config=configs['moe_config'],
        output_dir=configs['checkpoint_config'].output_dir,
    )


def rl_main(argv: Optional[List[str]] = None) -> List[dict]:
    return run_rl_configs(parse_rl_configs(argv))


if __name__ == '__main__':
    rl_main()

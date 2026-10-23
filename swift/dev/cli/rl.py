# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RlCliCompatConfig:
    """Deprecated RLHF aliases retained for command-line compatibility.

    ``response_length`` owning its spelling here suppresses the parser-level alias fold, so the
    value lands on this config and ``_apply_rl_compat`` folds it manually. ``--seq_kd`` is refused
    up front by ``reject_legacy_only_flags`` (its removal note lives in legacy_coverage), so it has
    no compat field here.
    """

    response_length: Optional[int] = None


def _apply_rl_compat(passed, generation_config, rl_config, rlhf_compat) -> None:
    if 'max_new_tokens' in passed and generation_config.max_new_tokens is not None:
        rl_config.max_completion_length = generation_config.max_new_tokens
    if rlhf_compat.response_length is not None:
        rl_config.max_completion_length = rlhf_compat.response_length


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
    # Every field name is unique across these Configs (``distill_temperature`` is the RL objective's,
    # ``max_new_tokens``/``temperature`` only on GenerationConfig), so no owner disambiguation is needed.
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
    # The online family rolls out on a separate vLLM/SGLang sampler over Ray and syncs weights into it;
    # on-policy generation must be backend-general (in-process ``model.generate`` is transformers-only).
    if rl_config.rl_type in {'grpo', 'ppo', 'rft', 'gkd', 'opsd', 'mopd'}:
        rollout_config.sampler_mode = rollout_config.sampler_mode or 'colocate'
        distributed_config.mode = 'ray'
    # The offline RLHF family runs mode='ray' with the policy and a full-param frozen reference as
    # separate actors; it rolls out nothing, so only the mode is forced.
    elif rl_config.rl_type in {'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'}:
        distributed_config.mode = 'ray'
    else:
        # Fail loudly: the two sets above are hand-kept in sync with RLConfig.rl_type's Literal, and a
        # value missing from both would otherwise silently fall back to mode='local', which RL never supports.
        raise ValueError(
            f'--rl_type {rl_config.rl_type!r} is not classified in parse_rl_configs. Every rl_type must '
            f"force distributed_config.mode='ray' (RL has no mode='local' path): add it to the online family "
            f'(rolls out on a separate vLLM sampler) or the offline RLHF family (mode only) above.')

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

    process_and_validate_configs(configs)
    # The recipes (five-segment mains): grpo, ppo, rft, the distillation family gkd/opsd/mopd,
    # and the offline RLHF family dpo/kto/cpo/orpo/simpo/rm; anything else fails loudly below.
    rl_type = configs['rl_config'].rl_type
    if rl_type == 'grpo':
        from swift.dev.recipe.grpo import run_grpo as run
    elif rl_type == 'ppo':
        from swift.dev.recipe.ppo import run_ppo as run
    elif rl_type == 'rft':
        from swift.dev.recipe.rft import run_rft as run
    elif rl_type in {'gkd', 'opsd', 'mopd'}:
        from swift.dev.recipe.distill import run_distill as run
    elif rl_type in {'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'}:
        # The offline RLHF family takes no rollout/generation configs.
        from swift.dev.recipe.rlhf import run_dpo as run
        return run(
            configs['model_config'],
            configs['template_config'],
            configs['dataset_config'],
            configs['train_config'],
            configs['distributed_config'],
            configs['checkpoint_config'],
            configs['rl_config'],
            reward_config=configs['reward_config'],
            tuner_config=configs['tuner_config'],
            logging_config=configs['logging_config'],
            quantize_config=configs['quantize_config'],
            megatron_config=configs['megatron_config'],
            moe_config=configs['moe_config'],
            output_dir=configs['checkpoint_config'].output_dir,
        )
    else:
        raise ValueError(
            f'--rl_type {rl_type!r} has no recipe. Supported: grpo / ppo / rft, the '
            'distillation family gkd / opsd / mopd, and the offline RLHF family '
            'dpo / kto / cpo / orpo / simpo / rm.')
    # The online recipes (grpo / ppo / rft / gkd / opsd / mopd) share one positional shape.
    return run(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        configs['train_config'],
        configs['distributed_config'],
        configs['checkpoint_config'],
        configs['rollout_config'],
        configs['rl_config'],
        reward_config=configs['reward_config'],
        tuner_config=configs['tuner_config'],
        generation_config=configs['generation_config'],
        logging_config=configs['logging_config'],
        quantize_config=configs['quantize_config'],
        megatron_config=configs['megatron_config'],
        moe_config=configs['moe_config'],
        output_dir=configs['checkpoint_config'].output_dir,
    )


def rl_main(argv: Optional[List[str]] = None) -> List[dict]:
    return run_rl_configs(parse_rl_configs(argv))


if __name__ == '__main__':
    rl_main()

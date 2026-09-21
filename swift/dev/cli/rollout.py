"""Standalone rollout-service CLI backed by the dev vLLM rollout engine."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def parse_rollout_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import parse_configs_strict, resolve_argv
    from swift.dev.cli.sft import _select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DeployConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        RLHFConfig,
        RolloutConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('rollout', effective_argv)
    classes = [ModelConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig, GenerationConfig,
               RolloutConfig, RLHFConfig, DeployConfig, RuntimeConfig]
    owners = {
        'max_new_tokens': GenerationConfig,
        'temperature': GenerationConfig,
        'top_k': GenerationConfig,
        'top_p': GenerationConfig,
        'repetition_penalty': GenerationConfig,
        'stop_words': GenerationConfig,
        'structured_outputs_regex': GenerationConfig,
        'reward_funcs': RLHFConfig,
        'reward_weights': RLHFConfig,
    }
    configs = parse_configs_strict(classes, effective_argv, command='swift rollout', field_owners=owners)
    names = ('model_config', 'template_config', 'dataset_config', 'checkpoint_config', 'tuner_config',
             'generation_config', 'rollout_config', 'rlhf_config', 'deploy_config', 'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = _select_tuner(result['tuner_config'])
    result['rollout_config'].use_vllm = True
    result['rollout_config'].vllm_mode = 'server'
    if result['deploy_config'].context_manager is not None:
        raise ValueError('`context_manager` has been removed; use a MultiTurnScheduler plugin instead.')
    # Kept here to make the backend contract explicit even if InferConfig later regains more choices.
    result['infer_config'] = InferConfig(infer_backend='vllm')
    return result


def rollout_main(argv: Optional[List[str]] = None) -> None:
    from swift.dev.cli.infer import _engine_args
    from swift.dev.cli.runtime import bootstrap_run, process_and_validate_configs
    from swift.dev.recipe import run_rollout

    configs = parse_rollout_configs(argv)
    process_and_validate_configs(configs)
    bootstrap_run(
        configs['model_config'], configs['checkpoint_config'], configs['dataset_config'], configs['tuner_config'],
        seed=configs['runtime_config'].seed, add_version=False, create_output_dir=False)
    return run_rollout(
        configs['model_config'],
        configs['template_config'],
        configs['rollout_config'],
        configs['rlhf_config'],
        configs['deploy_config'],
        configs['generation_config'],
        engine_args=_engine_args('vllm', configs['infer_config'], configs['rollout_config']),
    )


if __name__ == '__main__':
    rollout_main()

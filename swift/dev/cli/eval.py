"""Eval CLI backed by EvalScope and the dev deployment recipe."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional


def parse_eval_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.deploy import DeployCliConfig
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import parse_configs_strict, resolve_argv
    from swift.dev.cli.runtime import _parse_mapping
    from swift.dev.cli.sft import _select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DeployConfig,
        EvalConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        RolloutConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('eval', effective_argv)
    classes = [ModelConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig, GenerationConfig,
               RolloutConfig, InferConfig, DeployConfig, EvalConfig, DeployCliConfig, RuntimeConfig]
    owners = {
        'top_k': GenerationConfig,
        'top_p': GenerationConfig,
        'repetition_penalty': GenerationConfig,
        'stop_words': GenerationConfig,
        'structured_outputs_regex': GenerationConfig,
        'use_chat_template': EvalConfig,
    }
    configs = parse_configs_strict(classes, effective_argv, command='swift eval', field_owners=owners)
    names = ('model_config', 'template_config', 'dataset_config', 'checkpoint_config', 'tuner_config',
             'generation_config', 'rollout_config', 'infer_config', 'deploy_config', 'eval_config', 'cli_config',
             'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = _select_tuner(result['tuner_config'])
    eval_config = result['eval_config']
    eval_config.eval_dataset_args = _parse_mapping(eval_config.eval_dataset_args)
    eval_config.eval_generation_config = _parse_mapping(eval_config.eval_generation_config)
    eval_config.extra_eval_args = _parse_mapping(eval_config.extra_eval_args)
    eval_config.eval_output_dir = os.path.abspath(os.path.expanduser(eval_config.eval_output_dir))
    if eval_config.result_jsonl:
        eval_config.result_jsonl = os.path.abspath(os.path.expanduser(eval_config.result_jsonl))
    if result['infer_config'].infer_backend == 'pt':
        result['infer_config'].infer_backend = 'transformers'
    if eval_config.eval_url and result['cli_config'].merge_lora:
        raise ValueError('`--merge_lora` only applies when eval starts a local deployment; '
                         'it cannot modify the model served by `--eval_url`.')
    return result


def eval_main(argv: Optional[List[str]] = None):
    from swift.dev.cli.deploy import _adapter_mapping
    from swift.dev.cli.runtime import bootstrap_run, process_and_validate_configs
    from swift.dev.recipe import run_eval

    configs = parse_eval_configs(argv)
    process_and_validate_configs(configs)
    remote = bool(configs['eval_config'].eval_url)
    bootstrap_run(
        configs['model_config'], configs['checkpoint_config'], configs['dataset_config'], configs['tuner_config'],
        seed=configs['runtime_config'].seed, add_version=False, create_output_dir=False, resolve_model=not remote)
    adapters = configs['tuner_config'].adapters if configs['tuner_config'] is not None else []
    return run_eval(
        configs['model_config'],
        configs['template_config'],
        configs['generation_config'],
        configs['infer_config'],
        configs['rollout_config'],
        configs['deploy_config'],
        configs['eval_config'],
        adapter_mapping=_adapter_mapping(adapters),
        merge_lora=configs['cli_config'].merge_lora,
    )


if __name__ == '__main__':
    eval_main()

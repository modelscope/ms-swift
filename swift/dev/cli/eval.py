"""Eval CLI: parse/validate dev Configs and drive the in-process sampler evaluation recipe."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional


def parse_eval_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import parse_configs_strict, resolve_argv, select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        EvalConfig,
        InferConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        RolloutConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('eval', effective_argv)
    classes = [ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig,
               RolloutConfig, InferConfig, EvalConfig, QuantizeConfig, RuntimeConfig]
    configs = parse_configs_strict(classes, effective_argv, command='swift eval', load_args_default=True)
    names = ('model_config', 'plugin_config', 'template_config', 'dataset_config', 'checkpoint_config',
             'tuner_config', 'rollout_config', 'infer_config', 'eval_config', 'quantize_config', 'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = select_tuner(result['tuner_config'])
    eval_config = result['eval_config']
    eval_config.eval_output_dir = os.path.abspath(os.path.expanduser(eval_config.eval_output_dir))
    if eval_config.result_jsonl:
        eval_config.result_jsonl = os.path.abspath(os.path.expanduser(eval_config.result_jsonl))
    if result['infer_config'].sampler == 'pt':
        result['infer_config'].sampler = 'transformers'
    return result


def eval_main(argv: Optional[List[str]] = None):
    from swift.dev.builders import build_engine_args
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_eval

    configs = parse_eval_configs(argv)
    process_and_validate_configs(configs, add_version=False, create_output_dir=False, resolve_model=True)
    adapters = configs['tuner_config'].adapters if configs['tuner_config'] is not None else None
    backend = configs['infer_config'].sampler
    engine_args = build_engine_args(backend, configs['infer_config'], configs['rollout_config'])
    return run_eval(
        configs['model_config'],
        configs['template_config'],
        configs['eval_config'],
        backend=backend,
        engine_args=engine_args,
        adapters=adapters,
        quantize_config=configs['quantize_config'],
    )


if __name__ == '__main__':
    eval_main()

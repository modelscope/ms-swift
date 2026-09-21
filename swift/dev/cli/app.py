"""Gradio application CLI backed by the dev deployment recipe."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def parse_app_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.deploy import DeployCliConfig
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import parse_configs_strict, resolve_argv
    from swift.dev.cli.sft import _select_tuner
    from swift.dev.config import (
        AppConfig,
        CheckpointConfig,
        DatasetConfig,
        DeployConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        RolloutConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('app', effective_argv)
    classes = [ModelConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig, GenerationConfig,
               RolloutConfig, InferConfig, DeployConfig, AppConfig, DeployCliConfig, RuntimeConfig]
    owners = {
        'top_k': GenerationConfig,
        'top_p': GenerationConfig,
        'repetition_penalty': GenerationConfig,
        'stop_words': GenerationConfig,
        'structured_outputs_regex': GenerationConfig,
    }
    configs = parse_configs_strict(classes, effective_argv, command='swift app', field_owners=owners)
    names = ('model_config', 'template_config', 'dataset_config', 'checkpoint_config', 'tuner_config',
             'generation_config', 'rollout_config', 'infer_config', 'deploy_config', 'app_config', 'cli_config',
             'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = _select_tuner(result['tuner_config'])
    if result['infer_config'].infer_backend == 'pt':
        result['infer_config'].infer_backend = 'transformers'
    if result['deploy_config'].context_manager is not None:
        raise ValueError('`context_manager` has been removed; use a MultiTurnScheduler plugin instead.')
    if result['app_config'].base_url and result['cli_config'].merge_lora:
        raise ValueError('`--merge_lora` only applies when app starts a local deployment; '
                         'it cannot modify the model served by `--base_url`.')
    return result


def _derive_ui_defaults(configs: Dict[str, Any]) -> None:
    from swift.model import get_matched_model_meta
    from swift.template import get_template_meta
    from swift.utils import find_free_port

    app_config = configs['app_config']
    model_config = configs['model_config']
    template_config = configs['template_config']
    app_config.server_port = find_free_port(app_config.server_port)
    model_meta = get_matched_model_meta(model_config.model) if model_config.model else None
    if app_config.is_multimodal is None:
        app_config.is_multimodal = bool(model_meta and model_meta.is_multimodal)
    if template_config.system is None and model_meta is not None:
        try:
            template_config.system = get_template_meta(None, model_meta, template_config.template).default_system
        except ValueError:
            pass


def app_main(argv: Optional[List[str]] = None) -> None:
    from swift.dev.cli.deploy import _adapter_mapping
    from swift.dev.cli.runtime import bootstrap_run, process_and_validate_configs
    from swift.dev.recipe import run_app

    configs = parse_app_configs(argv)
    process_and_validate_configs(configs)
    _derive_ui_defaults(configs)
    remote = bool(configs['app_config'].base_url)
    bootstrap_run(
        configs['model_config'], configs['checkpoint_config'], configs['dataset_config'], configs['tuner_config'],
        seed=configs['runtime_config'].seed, add_version=False, create_output_dir=False, resolve_model=not remote)
    adapters = configs['tuner_config'].adapters if configs['tuner_config'] is not None else []
    return run_app(
        configs['model_config'],
        configs['template_config'],
        configs['generation_config'],
        configs['infer_config'],
        configs['rollout_config'],
        configs['deploy_config'],
        configs['app_config'],
        adapter_mapping=_adapter_mapping(adapters),
        merge_lora=configs['cli_config'].merge_lora,
    )


if __name__ == '__main__':
    app_main()

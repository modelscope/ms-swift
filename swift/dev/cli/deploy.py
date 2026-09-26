"""OpenAI-compatible deployment CLI backed by the dev serving recipe."""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def _adapter_mapping(adapters: List[str]) -> Dict[str, str]:
    mapping = {}
    for index, item in enumerate(adapters):
        if '=' in item:
            name, path = item.split('=', 1)
        else:
            path = item
            name = f'adapter-{index + 1}'
        mapping[name] = path
    return mapping


def parse_deploy_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import parse_configs_strict, resolve_argv, select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DeployConfig,
        GenerationConfig,
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
    reject_legacy_only_flags('deploy', effective_argv)
    classes = [ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig,
               GenerationConfig, RolloutConfig, InferConfig, DeployConfig, QuantizeConfig, RuntimeConfig]
    configs = parse_configs_strict(
        classes, effective_argv, command='swift deploy', load_args_default=True)
    names = ('model_config', 'plugin_config', 'template_config', 'dataset_config', 'checkpoint_config',
             'tuner_config', 'generation_config', 'rollout_config', 'infer_config', 'deploy_config',
             'quantize_config', 'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = select_tuner(result['tuner_config'])
    if result['infer_config'].infer_backend == 'pt':
        result['infer_config'].infer_backend = 'transformers'
    return result


def deploy_main(argv: Optional[List[str]] = None) -> None:
    from swift.dev.builders import build_engine_args
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_deploy

    configs = parse_deploy_configs(argv)
    process_and_validate_configs(configs, add_version=False, create_output_dir=False)
    tuner_config = configs['tuner_config']
    adapters = tuner_config.adapters if tuner_config is not None else []
    deploy = configs['deploy_config']
    if deploy.context_manager is not None:
        raise ValueError('`context_manager` has been removed; use a MultiTurnScheduler plugin instead.')
    return run_deploy(
        configs['model_config'],
        configs['template_config'],
        configs['generation_config'],
        backend=configs['infer_config'].infer_backend,
        engine_args=build_engine_args(configs['infer_config'].infer_backend, configs['infer_config'],
                                      configs['rollout_config']),
        adapter_mapping=_adapter_mapping(adapters),
        quantize_config=configs['quantize_config'],
        merge_lora=deploy.merge_lora,
        host=deploy.host,
        port=deploy.port,
        served_model_name=deploy.served_model_name,
        owned_by=deploy.owned_by,
        api_key=deploy.api_key,
        max_logprobs=deploy.max_logprobs,
        max_concurrency=deploy.max_concurrency,
        log_interval=deploy.log_interval,
        request_log_path=deploy.request_log_path,
        verbose=deploy.verbose,
        ssl_keyfile=deploy.ssl_keyfile,
        ssl_certfile=deploy.ssl_certfile,
        log_level=deploy.log_level,
    )


if __name__ == '__main__':
    deploy_main()

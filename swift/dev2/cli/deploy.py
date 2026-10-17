# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, List, Optional


def _adapter_mapping(adapters: List[str]) -> Dict[str, str]:
    from swift.dev2.config.process import split_served_name
    mapping = {}
    for index, item in enumerate(adapters):
        name, path = split_served_name(item)
        if name is None:
            name = f'adapter-{index + 1}'
        mapping[name] = path
    return mapping


def _strip_adapter_names(argv: List[str]):
    """Rewrite ``--adapters name=path`` to bare paths; return ``(argv, names)``.

    The checkpoint-args restore resolves each ``--adapters`` entry as a path/hub id (a literal
    ``name=path`` would 404), so the parser sees bare paths; names re-attach after parsing.
    """
    names: List[Optional[str]] = []
    out: List[str] = []
    from swift.dev2.config.process import split_served_name
    index = 0
    while index < len(argv):
        token = argv[index]
        inline = token.startswith('--adapters=')
        if token == '--adapters' or inline:
            if inline:
                name, path = split_served_name(token.split('=', 1)[1])
                names.append(name)
                out.append(f'--adapters={path}')
                index += 1
            else:
                out.append(token)
                index += 1
            # ``adapters`` is a List[str] (nargs='+'): consume the value tokens up to the next flag.
            while index < len(argv) and not argv[index].startswith('--'):
                name, path = split_served_name(argv[index])
                names.append(name)
                out.append(path)
                index += 1
            continue
        out.append(token)
        index += 1
    return out, names


def parse_deploy_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev2.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev2.cli.parser import normalize_sampler, parse_configs_strict, resolve_argv, select_tuner
    from swift.dev2.config import (
        CheckpointConfig,
        DatasetConfig,
        DeployConfig,
        DistributedConfig,
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
    effective_argv, adapter_names = _strip_adapter_names(effective_argv)
    classes = [ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig,
               TunerConfig, GenerationConfig, RolloutConfig, InferConfig, DeployConfig, QuantizeConfig,
               RuntimeConfig]
    # ``sampler_type`` (DeployConfig's twinkle-server sampler kind) and ``strict`` (DatasetConfig's
    # column-mapping gate) are each unique across these Configs, so no owner pinning is needed.
    configs = parse_configs_strict(
        classes, effective_argv, command='swift deploy', load_args_default=True)
    names = ('model_config', 'plugin_config', 'template_config', 'dataset_config', 'distributed_config',
             'checkpoint_config', 'tuner_config', 'generation_config', 'rollout_config', 'infer_config',
             'deploy_config', 'quantize_config', 'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = select_tuner(result['tuner_config'])
    # Re-attach the served names the parser never saw, onto the (possibly checkpoint-resolved) paths.
    tuner_config = result['tuner_config']
    if tuner_config is not None and adapter_names and len(adapter_names) == len(tuner_config.adapters):
        tuner_config.adapters = [f'{name}={path}' if name else path
                                 for name, path in zip(adapter_names, tuner_config.adapters)]
    normalize_sampler(result['infer_config'])
    return result


def deploy_main(argv: Optional[List[str]] = None) -> None:
    from swift.dev2.builders import build_engine_args
    from swift.dev2.config import process_and_validate_configs
    from swift.dev.recipe import run_deploy

    configs = parse_deploy_configs(argv)
    process_and_validate_configs(configs, add_version=False, create_output_dir=False)
    tuner_config = configs['tuner_config']
    adapters = tuner_config.adapters if tuner_config is not None else []
    deploy = configs['deploy_config']
    run_deploy(
        configs['model_config'],
        configs['template_config'],
        configs['generation_config'],
        backend=configs['infer_config'].sampler,
        engine_args=build_engine_args(configs['infer_config'].sampler, configs['infer_config'],
                                      configs['rollout_config']),
        adapter_mapping=_adapter_mapping(adapters),
        quantize_config=configs['quantize_config'],
        distributed_config=configs['distributed_config'],
        deploy_config=deploy,
    )


if __name__ == '__main__':
    deploy_main()

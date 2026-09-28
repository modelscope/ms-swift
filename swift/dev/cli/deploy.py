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


def _split_adapter_name(item: str):
    """``name=path`` -> ``(name, path)``; a bare path/hub id -> ``(None, item)``."""
    if '=' in item:
        name, path = item.split('=', 1)
        return name, path
    return None, item


def _strip_adapter_names(argv: List[str]):
    """Rewrite ``--adapters name=path`` to bare ``--adapters path``; return ``(argv, names)``.

    Deploy routes a named adapter as ``name=path`` (see :func:`_adapter_mapping`), but the shared
    checkpoint-args restore resolves every ``--adapters`` entry as a path/hub id and would try to
    download the literal ``name=path`` (a 404 on a string that is neither). Hand the parser the bare
    path so the restore and hub resolution work, and keep the index-aligned names to re-attach after
    parsing, so ``_adapter_mapping`` still yields the operator's served names.
    """
    names: List[Optional[str]] = []
    out: List[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        inline = token.startswith('--adapters=')
        if token == '--adapters' or inline:
            if inline:
                name, path = _split_adapter_name(token.split('=', 1)[1])
                names.append(name)
                out.append(f'--adapters={path}')
                index += 1
            else:
                out.append(token)
                index += 1
            # ``adapters`` is a List[str] (nargs='+'): consume the value tokens up to the next flag.
            while index < len(argv) and not argv[index].startswith('--'):
                name, path = _split_adapter_name(argv[index])
                names.append(name)
                out.append(path)
                index += 1
            continue
        out.append(token)
        index += 1
    return out, names


def parse_deploy_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import parse_configs_strict, resolve_argv, select_tuner
    from swift.dev.config import (
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
    # ``sampler_type`` is spelled by two Configs: InferConfig's is a synthesis label ('sample'/'distill')
    # that means nothing when serving, while DeployConfig's is the twinkle-server sampler kind
    # ('vllm'/'vllm_async'/'sglang_async'/...) that ``_validate`` guards and ``build_server_config`` pins.
    # Pin the CLI spelling to DeployConfig so ``swift deploy --sampler_type vllm_async`` reaches the field
    # that actually drives the deployment instead of being rejected by InferConfig's narrower choices.
    owners = {'sampler_type': DeployConfig}
    configs = parse_configs_strict(
        classes, effective_argv, command='swift deploy', field_owners=owners, load_args_default=True)
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
    return run_deploy(
        configs['model_config'],
        configs['template_config'],
        configs['generation_config'],
        backend=configs['infer_config'].infer_backend,
        engine_args=build_engine_args(configs['infer_config'].infer_backend, configs['infer_config'],
                                      configs['rollout_config']),
        adapter_mapping=_adapter_mapping(adapters),
        quantize_config=configs['quantize_config'],
        distributed_config=configs['distributed_config'],
        deploy_config=deploy,
    )


if __name__ == '__main__':
    deploy_main()

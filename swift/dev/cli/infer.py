"""Inference CLI backed only by dev Configs and recipes."""
from __future__ import annotations
import datetime as dt
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class InferCliConfig:
    eval_human: bool = False
    merge_lora: bool = False
    num_samples: int = 1
    multi_round: bool = True


def _engine_args(backend: str, infer_config, rollout_config) -> Dict[str, Any]:
    from swift.dev.cli.runtime import _parse_mapping

    if backend == 'pt':
        backend = 'transformers'
    if backend == 'transformers':
        return {'max_batch_size': infer_config.max_batch_size}
    prefix = f'{backend}_'
    result = {}
    for name, value in asdict(rollout_config).items():
        if not name.startswith(prefix) or value is None:
            continue
        key = name[len(prefix):]
        if key in {'engine_kwargs', 'mode', 'server_base_url', 'server_host', 'server_port', 'server_timeout',
                   'server_group_port', 'server_pass_dataset'}:
            continue
        result[key] = value
    extra = getattr(rollout_config, f'{prefix}engine_kwargs', None)
    result.update(_parse_mapping(extra))
    for key in ('limit_mm_per_prompt', 'speculative_config'):
        if key in result and isinstance(result[key], str):
            result[key] = _parse_mapping(result[key])
    return result


def parse_infer_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import parse_configs_strict, resolve_argv
    from swift.dev.cli.sft import _select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        RolloutConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('infer', effective_argv)
    classes = [ModelConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig, TunerConfig,
               GenerationConfig, RolloutConfig, InferConfig, InferCliConfig, RuntimeConfig]
    owners = {
        'top_k': GenerationConfig,
        'top_p': GenerationConfig,
        'repetition_penalty': GenerationConfig,
        'stop_words': GenerationConfig,
        'structured_outputs_regex': GenerationConfig,
    }
    configs = parse_configs_strict(classes, effective_argv, command='swift infer', field_owners=owners)
    names = ('model_config', 'template_config', 'dataset_config', 'distributed_config', 'checkpoint_config',
             'tuner_config', 'generation_config', 'rollout_config', 'infer_config', 'cli_config', 'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = _select_tuner(result['tuner_config'])
    if result['infer_config'].infer_backend == 'pt':
        result['infer_config'].infer_backend = 'transformers'
    has_dataset = bool(result['dataset_config'].dataset or result['dataset_config'].val_dataset)
    if result['generation_config'].stream is None:
        result['generation_config'].stream = not has_dataset
    if result['generation_config'].stream and result['generation_config'].num_beams != 1:
        result['generation_config'].stream = False
    return result


def _derive_result_path(configs: Dict[str, Any]) -> None:
    infer_config = configs['infer_config']
    dataset_config = configs['dataset_config']
    if infer_config.result_path:
        infer_config.result_path = os.path.abspath(os.path.expanduser(infer_config.result_path))
    elif dataset_config.dataset or dataset_config.val_dataset:
        model = (configs['model_config'].model or 'model').rstrip('/')
        model_suffix = os.path.basename(model)
        timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        infer_config.result_path = os.path.abspath(
            os.path.join('result', model_suffix, 'infer_result', f'{timestamp}.jsonl'))


def infer_main(argv: Optional[List[str]] = None):
    from swift.dev.cli.runtime import bootstrap_run, process_and_validate_configs
    from swift.dev.recipe import infer_cli, run_infer

    configs = parse_infer_configs(argv)
    process_and_validate_configs(configs)
    bootstrap_run(
        configs['model_config'], configs['checkpoint_config'], configs['dataset_config'], configs['tuner_config'],
        seed=configs['runtime_config'].seed, add_version=False, create_output_dir=False)
    _derive_result_path(configs)
    backend = configs['infer_config'].infer_backend
    engine_args = _engine_args(backend, configs['infer_config'], configs['rollout_config'])
    adapters = configs['tuner_config'].adapters if configs['tuner_config'] is not None else None
    if configs['cli_config'].eval_human or not (configs['dataset_config'].dataset
                                                or configs['dataset_config'].val_dataset):
        return infer_cli(
            configs['model_config'],
            configs['template_config'],
            configs['generation_config'],
            backend=backend,
            engine_args=engine_args,
            adapters=adapters,
            multi_round=configs['cli_config'].multi_round,
        )
    return run_infer(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        configs['generation_config'],
        backend=backend,
        engine_args=engine_args,
        distributed_config=configs['distributed_config'],
        tuner_config=configs['tuner_config'],
        adapters=adapters,
        merge_lora=configs['cli_config'].merge_lora,
        num_samples=configs['cli_config'].num_samples,
        max_rows=configs['infer_config'].val_dataset_sample,
        split_dataset_ratio=configs['dataset_config'].split_dataset_ratio,
        output_path=configs['infer_config'].result_path,
        write_batch_size=configs['infer_config'].write_batch_size,
        metric=configs['infer_config'].metric,
        strict=configs['dataset_config'].strict,
    )


if __name__ == '__main__':
    infer_main()

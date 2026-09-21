"""Best-of-n and distillation sampling CLI."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional


def parse_sample_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import flag_names, parse_configs_strict, resolve_argv
    from swift.dev.cli.runtime import _parse_mapping
    from swift.dev.cli.sft import _select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        RLHFConfig,
        RolloutConfig,
        RuntimeConfig,
        SamplingConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('sample', effective_argv)
    classes = [ModelConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig, TunerConfig,
               GenerationConfig, RolloutConfig, InferConfig, SamplingConfig, RLHFConfig, RuntimeConfig]
    owners = {
        'strict': SamplingConfig,
        'temperature': GenerationConfig,
        'top_k': GenerationConfig,
        'top_p': GenerationConfig,
        'repetition_penalty': GenerationConfig,
        'stop_words': GenerationConfig,
        'structured_outputs_regex': GenerationConfig,
        'reward_funcs': SamplingConfig,
        'reward_weights': SamplingConfig,
    }
    configs = parse_configs_strict(classes, effective_argv, command='swift sample', field_owners=owners)
    names = ('model_config', 'template_config', 'dataset_config', 'distributed_config', 'checkpoint_config',
             'tuner_config', 'generation_config', 'rollout_config', 'infer_config', 'sampling_config', 'reward_config',
             'runtime_config')
    result = dict(zip(names, configs))
    sampling = result['sampling_config']
    passed = flag_names(effective_argv)
    if sampling.sampler_engine == 'pt':
        sampling.sampler_engine = 'transformers'
    if sampling.num_sampling_batch_size is not None:
        sampling.batch_size = sampling.num_sampling_batch_size
    if sampling.num_sampling_batches is not None:
        sampling.max_batches = sampling.num_sampling_batches
    if sampling.prm_threshold is not None and 'reward_threshold' not in passed:
        sampling.reward_threshold = sampling.prm_threshold
    if sampling.output_file is None:
        sampling.output_file = datetime.now().strftime('%Y-%m-%d-%H-%M-%S.jsonl')
    if sampling.data_range is not None:
        sampling.data_range = tuple(sampling.data_range)
    if 'padding_side' not in passed:
        result['template_config'].padding_side = 'left'
    sampling.reward_config = result['reward_config']
    sampling.engine_kwargs = _parse_mapping(sampling.engine_kwargs)
    result['tuner_config'] = _select_tuner(result['tuner_config'])
    return result


def sample_main(argv: Optional[List[str]] = None) -> str:
    from swift.dev.cli.infer import _engine_args
    from swift.dev.cli.runtime import bootstrap_run, process_and_validate_configs
    from swift.dev.recipe import run_sampling

    configs = parse_sample_configs(argv)
    process_and_validate_configs(configs)
    sampling = configs['sampling_config']
    bootstrap_run(
        configs['model_config'], configs['checkpoint_config'], configs['dataset_config'], configs['tuner_config'],
        seed=configs['runtime_config'].seed, add_version=False,
        resolve_model=sampling.sampler_engine not in {'client', 'no'})
    backend = sampling.sampler_engine
    engine_args = _engine_args(backend, configs['infer_config'], configs['rollout_config'])
    engine_args.update(sampling.engine_kwargs or {})
    adapters = configs['tuner_config'].adapters if configs['tuner_config'] is not None else None
    return run_sampling(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        sampling,
        configs['generation_config'],
        backend=backend,
        engine_args=engine_args,
        distributed_config=configs['distributed_config'],
        adapters=adapters,
        output_dir=configs['checkpoint_config'].output_dir,
    )


if __name__ == '__main__':
    sample_main()

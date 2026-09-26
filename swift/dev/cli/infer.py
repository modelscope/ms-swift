"""Inference CLI backed only by dev Configs and recipes."""
from __future__ import annotations
import datetime as dt
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class InferCliConfig:
    eval_human: bool = False
    merge_lora: bool = False
    num_samples: int = 1
    multi_round: bool = True


def parse_infer_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import flag_names, parse_configs_strict, resolve_argv, select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        RLHFConfig,
        RolloutConfig,
        RuntimeConfig,
        SamplingConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('infer', effective_argv)
    # infer absorbs the whole sampling surface: SamplingConfig drives best-of-n / reward / token-dump and
    # RLHFConfig is the multi-turn carrier (max_turns, tools). The duplicate spellings these add are each
    # pinned to one owner so a flag never lands on the wrong Config.
    classes = [ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig,
               TunerConfig, GenerationConfig, RolloutConfig, InferConfig, SamplingConfig, RLHFConfig,
               QuantizeConfig, InferCliConfig, RuntimeConfig]
    owners = {
        'strict': SamplingConfig,
        'temperature': GenerationConfig,
        'reward_funcs': SamplingConfig,
        'reward_weights': SamplingConfig,
    }
    configs = parse_configs_strict(
        classes, effective_argv, command='swift infer', field_owners=owners, load_args_default=True)
    names = ('model_config', 'plugin_config', 'template_config', 'dataset_config', 'distributed_config',
             'checkpoint_config', 'tuner_config', 'generation_config', 'rollout_config', 'infer_config',
             'sampling_config', 'reward_config', 'quantize_config', 'cli_config', 'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = select_tuner(result['tuner_config'])
    sampling = result['sampling_config']
    infer_config = result['infer_config']
    if infer_config.infer_backend == 'pt':
        infer_config.infer_backend = 'transformers'
    passed = flag_names(effective_argv)
    # Backend authority: an explicit --sampler_engine wins (it is the richer Literal that can name the
    # message-only 'client'/'no' backends infer_backend cannot); otherwise it follows --infer_backend.
    if sampling.sampler_engine == 'pt':
        sampling.sampler_engine = 'transformers'
    if 'sampler_engine' not in passed:
        sampling.sampler_engine = infer_config.infer_backend
    # Legacy sampling spellings folded into their canonical fields, mirroring the old sample CLI.
    if sampling.num_sampling_batch_size is not None:
        sampling.batch_size = sampling.num_sampling_batch_size
    if sampling.num_sampling_batches is not None:
        sampling.max_batches = sampling.num_sampling_batches
    if sampling.prm_threshold is not None and 'reward_threshold' not in passed:
        sampling.reward_threshold = sampling.prm_threshold
    if 'padding_side' not in passed:
        result['template_config'].padding_side = 'left'
    # The RLHFConfig doubles as the reward-hyperparameter carrier and the multi-turn config, as in sample.
    sampling.reward_config = result['reward_config']
    result['multi_turn_config'] = result['reward_config']
    # num_samples (plain-infer completions per prompt) and num_return_sequences (the best-of-n group) are
    # one knob under two names; an explicit spelling drives the other so the two modes agree.
    if 'num_samples' in passed:
        sampling.num_return_sequences = result['cli_config'].num_samples
    if 'num_return_sequences' in passed:
        result['cli_config'].num_samples = sampling.num_return_sequences
    has_dataset = bool(result['dataset_config'].dataset or result['dataset_config'].val_dataset)
    if result['generation_config'].stream is None:
        result['generation_config'].stream = not has_dataset
    if result['generation_config'].stream and result['generation_config'].num_beams != 1:
        result['generation_config'].stream = False
    return result


def _derive_result_path(configs: Dict[str, Any]) -> None:
    infer_config = configs['infer_config']
    dataset_config = configs['dataset_config']
    sampling = configs.get('sampling_config')
    if infer_config.result_path:
        infer_config.result_path = os.path.abspath(os.path.expanduser(infer_config.result_path))
    elif sampling is not None and sampling.output_file:
        # An explicit --output_file (the sampling spelling) still resolves to one absolute result_path, so
        # the sampling branch can treat result_path as the single truth for where the jsonl is written.
        output_dir = configs['checkpoint_config'].output_dir
        infer_config.result_path = os.path.abspath(os.path.join(output_dir, sampling.output_file))
    elif dataset_config.dataset or dataset_config.val_dataset:
        model = (configs['model_config'].model or 'model').rstrip('/')
        model_suffix = os.path.basename(model)
        timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        infer_config.result_path = os.path.abspath(
            os.path.join('result', model_suffix, 'infer_result', f'{timestamp}.jsonl'))


def _is_sampling_mode(sampling: Any) -> bool:
    """Whether a dataset run goes through best-of-n sampling rather than plain inference.

    Any sampling-only intent -- reward scoring, distillation, token dumping, resume, a candidate cache, or
    a message-only backend run_infer cannot drive -- routes to run_sampling; otherwise it is plain infer.
    """
    return bool(sampling.reward_funcs or sampling.prm_funcs or sampling.sampler_type == 'distill'
                or sampling.save_rollout_tokens or sampling.resume or sampling.cache_files
                or sampling.sampler_engine in {'client', 'no'})


def infer_main(argv: Optional[List[str]] = None):
    from swift.dev.builders import build_engine_args
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import infer_cli, run_infer, run_sampling

    configs = parse_infer_configs(argv)
    sampling = configs['sampling_config']
    cli_config = configs['cli_config']
    has_dataset = bool(configs['dataset_config'].dataset or configs['dataset_config'].val_dataset)
    interactive = cli_config.eval_human or not has_dataset
    sampling_mode = not interactive and _is_sampling_mode(sampling)
    # A message-only sampling backend loads no local weights, so skip model resolution for it (as sample did).
    resolve_model = not (sampling_mode and sampling.sampler_engine in {'client', 'no'})
    process_and_validate_configs(
        configs, add_version=False, create_output_dir=False, resolve_model=resolve_model)
    _derive_result_path(configs)
    adapters = configs['tuner_config'].adapters if configs['tuner_config'] is not None else None
    # Sampling honors an explicit --sampler_engine (which may be client/no); the other paths run a local
    # engine and stay on infer_backend.
    backend = sampling.sampler_engine if sampling_mode else configs['infer_config'].infer_backend
    engine_args = build_engine_args(backend, configs['infer_config'], configs['rollout_config'])
    if sampling_mode:
        engine_args.update(sampling.engine_kwargs or {})

    if interactive:
        return infer_cli(
            configs['model_config'],
            configs['template_config'],
            configs['generation_config'],
            backend=backend,
            engine_args=engine_args,
            adapters=adapters,
            quantize_config=configs['quantize_config'],
            multi_round=cli_config.multi_round,
            rollout_config=configs['rollout_config'],
            multi_turn_config=configs['multi_turn_config'],
        )

    if sampling_mode:
        # result_path is the single output truth: run_sampling writes output_dir/output_file, so point both
        # halves at result_path's split and its checkpoint/token sidecars land beside the jsonl.
        result_path = configs['infer_config'].result_path
        sampling.output_file = os.path.basename(result_path)
        output_dir = os.path.dirname(result_path) or configs['checkpoint_config'].output_dir
        return run_sampling(
            configs['model_config'],
            configs['template_config'],
            configs['dataset_config'],
            sampling,
            configs['generation_config'],
            multi_turn_config=configs['multi_turn_config'],
            backend=backend,
            engine_args=engine_args,
            distributed_config=configs['distributed_config'],
            adapters=adapters,
            quantize_config=configs['quantize_config'],
            plugin_config=configs['plugin_config'],
            rollout_config=configs['rollout_config'],
            output_dir=output_dir,
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
        quantize_config=configs['quantize_config'],
        plugin_config=configs['plugin_config'],
        adapters=adapters,
        merge_lora=cli_config.merge_lora,
        num_samples=cli_config.num_samples,
        max_rows=configs['infer_config'].val_dataset_sample,
        split_dataset_ratio=configs['dataset_config'].split_dataset_ratio,
        output_path=configs['infer_config'].result_path,
        write_batch_size=configs['infer_config'].write_batch_size,
        metric=configs['infer_config'].metric,
        strict=sampling.strict,
    )


if __name__ == '__main__':
    infer_main()

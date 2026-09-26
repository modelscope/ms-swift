"""Inference CLI backed only by dev Configs and recipes."""
from __future__ import annotations
import datetime as dt
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from swift.dev.utils.logger import get_logger

logger = get_logger()


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
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('infer', effective_argv)
    # infer absorbs the whole synthesis surface: InferConfig now carries the best-of-n / reward / token-dump
    # knobs, and RLHFConfig is the reward-func and multi-turn carrier (reward_funcs, max_turns, tools). The
    # duplicate spellings these add are each pinned to one owner so a flag never lands on the wrong Config.
    classes = [ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig,
               TunerConfig, GenerationConfig, RolloutConfig, InferConfig, RLHFConfig,
               QuantizeConfig, InferCliConfig, RuntimeConfig]
    owners = {
        'strict': InferConfig,
        'temperature': GenerationConfig,
    }
    configs = parse_configs_strict(
        classes, effective_argv, command='swift infer', field_owners=owners, load_args_default=True)
    names = ('model_config', 'plugin_config', 'template_config', 'dataset_config', 'distributed_config',
             'checkpoint_config', 'tuner_config', 'generation_config', 'rollout_config', 'infer_config',
             'reward_config', 'quantize_config', 'cli_config', 'runtime_config')
    result = dict(zip(names, configs))
    result['tuner_config'] = select_tuner(result['tuner_config'])
    infer_config = result['infer_config']
    if infer_config.infer_backend == 'pt':
        infer_config.infer_backend = 'transformers'
    passed = flag_names(effective_argv)
    # Legacy sampling spellings folded into their canonical InferConfig fields, mirroring the old sample CLI.
    if infer_config.num_sampling_batch_size is not None:
        infer_config.batch_size = infer_config.num_sampling_batch_size
    if infer_config.num_sampling_batches is not None:
        infer_config.max_batches = infer_config.num_sampling_batches
    if infer_config.prm_threshold is not None and 'reward_threshold' not in passed:
        infer_config.reward_threshold = infer_config.prm_threshold
    if 'padding_side' not in passed:
        result['template_config'].padding_side = 'left'
    # The RLHFConfig doubles as the reward-hyperparameter carrier (reward_funcs / reward_weights) and the
    # multi-turn config; synthesis reads it directly rather than through an InferConfig field.
    result['multi_turn_config'] = result['reward_config']
    # num_samples (the CLI spelling) and num_return_sequences (the InferConfig field) are one knob under
    # two names; an explicit value for either drives the other so both entry points agree. Nothing else is
    # derived here -- how many candidates to draw, whether to score them, and how to store them
    # (output_format) each keep their own Config default and bend only to the flags the user passes.
    if 'num_samples' in passed:
        infer_config.num_return_sequences = result['cli_config'].num_samples
    if 'num_return_sequences' in passed:
        result['cli_config'].num_samples = infer_config.num_return_sequences
    if infer_config.output_format == 'all':
        # The best-of-n ranking knobs only shape 'dpo' output; under 'all' every candidate is stored as-is,
        # so a threshold set here would be silently ignored. Say so instead of dropping it on the floor.
        ignored = [name for name, value in (('reward_threshold', infer_config.reward_threshold),
                                            ('easy_query_threshold', infer_config.easy_query_threshold))
                   if value is not None]
        if 'n_best_to_keep' in passed:
            ignored.append('n_best_to_keep')
        if ignored:
            logger.warning("output_format='all' ignores %s; they only shape the best-of-n ranking under "
                           "output_format='dpo'. Switch to 'dpo' to apply them.", ', '.join(ignored))
    has_dataset = bool(result['dataset_config'].dataset or result['dataset_config'].val_dataset)
    if result['generation_config'].stream is None:
        result['generation_config'].stream = not has_dataset
    if result['generation_config'].stream and result['generation_config'].num_beams != 1:
        result['generation_config'].stream = False
    return result


def _derive_result_path(configs: Dict[str, Any]) -> None:
    infer_config = configs['infer_config']
    dataset_config = configs['dataset_config']
    # result_path is the single output truth for both plain infer and synthesis: an explicit --result_path
    # wins, otherwise a dataset run falls back to a timestamped path under result/<model>/infer_result.
    if infer_config.result_path:
        infer_config.result_path = os.path.abspath(os.path.expanduser(infer_config.result_path))
    elif dataset_config.dataset or dataset_config.val_dataset:
        model = (configs['model_config'].model or 'model').rstrip('/')
        model_suffix = os.path.basename(model)
        timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        infer_config.result_path = os.path.abspath(
            os.path.join('result', model_suffix, 'infer_result', f'{timestamp}.jsonl'))


def _interactive_dp_width(distributed_config: Any) -> int:
    """How many data-parallel drivers an interactive run would have.

    Under ``mode='ray'`` the recipe runs once on a single driver and DP is expressed inside the sampler's
    device mesh, so the width is that mesh's data world size (1 when no DP mesh is built). Under local
    (torchrun) mode every rank runs this same recipe, so the width is the torchrun world size.
    """
    if distributed_config is not None and distributed_config.mode == 'ray':
        from swift.dev.builders import build_device_mesh_if_dp
        mesh = build_device_mesh_if_dp(distributed_config)
        return int(getattr(mesh, 'data_world_size', 1)) if mesh is not None else 1
    return max(1, int(os.getenv('WORLD_SIZE') or 1))


def _guard_interactive(distributed_config: Any, task_type: Optional[str]) -> None:
    """Reject interactive REPL runs that cannot drive a single turn-by-turn conversation.

    Two cases, matching legacy's ``_init_ddp`` assertion that DDP forbids ``eval_human``/``stream``:

    * A non-generative ``task_type`` (pooling / reranker) scores a forward pass; the REPL only generates
      text, so it would silently produce the wrong thing. Score those over a dataset instead.
    * More than one data-parallel driver: under torchrun every rank would open a competing REPL, and a
      DP>1 sampler shards the batch across ranks. A single DP rank is fine, so ray with dp=1 (tp=N) --
      one driver, tensor parallelism inside the engine -- is allowed.
    """
    from swift.dev.builders import is_pooling_task
    task_type = task_type or 'causal_lm'
    if is_pooling_task(task_type) or task_type == 'generative_reranker':
        raise ValueError(
            f'task_type={task_type!r} scores a forward pass, which the interactive REPL cannot drive (it '
            'only generates text). Run over a dataset (--dataset / --val_dataset) to score it.')
    dp = _interactive_dp_width(distributed_config)
    if dp > 1:
        raise ValueError(
            f'The interactive REPL needs a single data-parallel driver, but this run has dp={dp}: under '
            'torchrun every rank would open its own REPL, and a DP>1 sampler shards the batch across '
            'ranks. Run with dp=1 (tensor parallelism is fine), or infer over a dataset.')


def infer_main(argv: Optional[List[str]] = None):
    from swift.dev.builders import build_engine_args
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import infer_cli, run_infer

    configs = parse_infer_configs(argv)
    infer_config = configs['infer_config']
    rlhf_config = configs['reward_config']
    cli_config = configs['cli_config']
    has_dataset = bool(configs['dataset_config'].dataset or configs['dataset_config'].val_dataset)
    interactive = cli_config.eval_human or not has_dataset
    if interactive:
        _guard_interactive(configs['distributed_config'], configs['model_config'].task_type)
    # A message-only backend ('client'/'no') loads no local weights, so a dataset run skips model
    # resolution. Interactive still resolves: the REPL builds its chat template from the model even when
    # the completions come from a remote client.
    resolve_model = interactive or infer_config.infer_backend not in {'client', 'no'}
    process_and_validate_configs(
        configs, add_version=False, create_output_dir=False, resolve_model=resolve_model)
    _derive_result_path(configs)
    adapters = configs['tuner_config'].adapters if configs['tuner_config'] is not None else None
    # backend is unified on infer_backend, whose Literal names the message-only 'client'/'no' too.
    backend = infer_config.infer_backend
    engine_args = build_engine_args(backend, infer_config, configs['rollout_config'])
    # engine_kwargs is a generic escape hatch into the engine args, not a synthesis-only knob: a plain
    # inference run tunes its engine the same way, so it is merged regardless of intent.
    if infer_config.engine_kwargs:
        engine_args.update(infer_config.engine_kwargs)

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

    # result_path is the single output truth: run_infer splits a 'dpo'/resumed run's path into
    # output_dir/output_file, so its checkpoint/token sidecars land beside the jsonl.
    return run_infer(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        infer_config,
        configs['generation_config'],
        rlhf_config=rlhf_config,
        rollout_config=configs['rollout_config'],
        backend=backend,
        engine_args=engine_args,
        distributed_config=configs['distributed_config'],
        tuner_config=configs['tuner_config'],
        adapters=adapters,
        quantize_config=configs['quantize_config'],
        plugin_config=configs['plugin_config'],
        merge_lora=cli_config.merge_lora,
        max_rows=infer_config.val_dataset_sample,
        split_dataset_ratio=configs['dataset_config'].split_dataset_ratio,
        output_path=infer_config.result_path,
    )


if __name__ == '__main__':
    infer_main()

"""SFT CLI: argv -> dev Configs -> run_sft.

dev counterpart of legacy ``swift sft``. The Config dataclasses ARE the argument surface: argv is
parsed straight into them (see ``swift.dev.cli.parser``), so there is no legacy ``SftArguments`` bridge
and no same-name copy step. The eight Configs SFT needs share no field name, so a single flat argparse
namespace has no collisions.

``LoggingConfig`` is parsed alongside the seven core Configs and reaches the common dev tracker used by
both transformers and Megatron training loops.
``GenerationConfig`` stays outside the parse set because its training-time fields are already owned by
``TrainConfig``; unsupported generation flags land in ``remaining`` and are refused.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

#: Megatron spellings of hyperparameters that TrainConfig also carries under an HF name (lr vs
#: learning_rate, train_iters vs max_steps, ...). All Optional, so "is not None" means the user set it.
#: On the HF surface (backend != megatron) these are read by nobody -- the transformers path uses the HF
#: names -- so a value here would be silently ignored. That is the one footgun self-parse re-opens (the
#: legacy bridge caught it by refusing the whole Megatron surface), so it is guarded explicitly below.
_MEGATRON_ONLY_HPARAMS = ('lr', 'train_iters', 'micro_batch_size', 'global_batch_size', 'lr_warmup_fraction',
                          'lr_warmup_iters')

# Exhaustive classification of the legacy SftArguments fields which have no dev Config field. Two
# common precision aliases are handled by SftCliCompatConfig below; every remaining field is rejected
# with its category rather than disappearing in a generic same-name copy.
LEGACY_ONLY_CLASSIFICATION: Dict[str, Tuple[str, ...]] = {
    'unsupported_trainer': (
        'optim_target_modules', 'log_on_each_node', 'include_num_input_tokens_seen', 'log_level',
        'log_level_replica', 'project', 'trackio_space_id', 'trackio_bucket_id', 'trackio_static_space_id',
        'eval_do_concat_batches', 'eval_use_gather_object', 'include_for_metrics', 'batch_eval_metrics',
        'enable_jit_checkpoint', 'restore_callback_states_from_checkpoint', 'use_cpu', 'parallelism_config',
        'train_sampling_strategy', 'length_column_name', 'debug', 'skip_memory_metrics', 'do_train', 'do_predict',
        'sortish_sampler', 'ignore_args_error', 'use_swift_lora'),
    'unsupported_tuner': (
        'lisa_activated_layers', 'lisa_step_interval', 'lora_ga_batch_size', 'lora_ga_iters', 'lora_ga_max_length',
        'lora_ga_direction', 'lora_ga_scale', 'lora_ga_stable_gamma', 'fourier_n_frequency', 'fourier_scaling',
        'boft_block_size', 'boft_block_num', 'boft_n_butterfly_factor', 'boft_dropout', 'vera_rank',
        'vera_projection_prng_key', 'vera_dropout', 'vera_d_initial', 'adapter_act', 'adapter_length',
        'llamapro_num_new_blocks', 'llamapro_num_groups', 'reft_layer_key', 'reft_layers', 'reft_rank',
        'reft_intervention_type', 'reft_args'),
}


@dataclass
class SftCliCompatConfig:
    """Useful legacy precision aliases which do have an exact dev representation."""

    bf16: Optional[bool] = None
    fp16: Optional[bool] = None


def _flag_names(argv: List[str]) -> set:
    return {token[2:].split('=', 1)[0].replace('-', '_') for token in argv if token.startswith('--')}


def _reject_unconsumed_flags(passed_flags: set) -> None:
    from swift.dev.config import GenerationConfig, QuantizeConfig

    if 'optimizer' in passed_flags:
        raise NotImplementedError('`--optimizer` selects the Megatron optimizer type and is not consumed by the '
                                  'transformers SFT backend. Use `--optim` for the transformers optimizer.')
    for category, names in LEGACY_ONLY_CLASSIFICATION.items():
        unsupported = sorted(passed_flags.intersection(names))
        if unsupported:
            raise NotImplementedError(f'{unsupported} are legacy {category} flags with no dev pipeline consumer. '
                                      'They are refused rather than silently ignored.')

    # These Configs exist, but the SFT recipe has no consumer for them yet. Do not add them to the
    # parser merely to make the flags look supported: that would turn an explicit gap into a no-op.
    import dataclasses
    unconsumed_config_fields = {
        'quantization': {f.name for f in dataclasses.fields(QuantizeConfig)},
        'generation': {f.name for f in dataclasses.fields(GenerationConfig)} - {'max_new_tokens', 'temperature'},
    }
    for category, names in unconsumed_config_fields.items():
        unsupported = sorted(passed_flags.intersection(names))
        if unsupported:
            raise NotImplementedError(f'{unsupported} belong to {category} Config but the dev SFT recipe has no '
                                      'consumer for them yet. They are refused rather than silently ignored.')


def _apply_precision_aliases(model_config, compat: SftCliCompatConfig) -> None:
    if compat.fp16 and compat.bf16:
        raise ValueError('--fp16 and --bf16 are mutually exclusive.')
    flag_name = 'bf16' if compat.bf16 else ('fp16' if compat.fp16 else None)
    flag_dtype = 'bfloat16' if compat.bf16 else ('float16' if compat.fp16 else None)
    if flag_dtype is None:
        return
    if model_config.torch_dtype is not None and model_config.torch_dtype != flag_dtype:
        raise ValueError(f'--torch_dtype {model_config.torch_dtype!r} conflicts with --{flag_name}.')
    model_config.torch_dtype = flag_dtype


def _select_tuner(tuner: 'TunerConfig') -> Optional['TunerConfig']:
    return None if tuner.tuner_type == 'full' else tuner


def parse_sft_configs(
    argv: Optional[List[str]] = None,
) -> Tuple['ModelConfig', 'TemplateConfig', 'DatasetConfig', 'TrainConfig', 'DistributedConfig', 'CheckpointConfig',
           'LoggingConfig', Optional['TunerConfig']]:
    """Parse argv into the Configs run_sft consumes; dispatch the tuner by ``tuner_type``.

    Returns tuner_config=None for full-parameter training (tuner_type='full') and a filled TunerConfig
    for 'lora'. Any other tuner_type is refused rather than silently building LoRA.
    """
    from swift.dev.cli.parser import parse_configs, resolve_argv
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    _reject_unconsumed_flags(_flag_names(effective_argv))

    classes = [
        ModelConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig, LoggingConfig,
        TunerConfig, SftCliCompatConfig
    ]
    configs, remaining = parse_configs(classes, effective_argv)
    if remaining:
        raise ValueError(f'Unrecognized arguments: {remaining}. The dev SFT CLI parses the Config surface '
                         'directly; a flag with no matching Config field is refused rather than dropped.')
    (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
     logging_config, tuner, compat) = configs

    _apply_precision_aliases(model_config, compat)

    from swift.dev.builders.model import is_megatron_backend
    if not is_megatron_backend(distributed_config):
        set_megatron = [name for name in _MEGATRON_ONLY_HPARAMS if getattr(train_config, name) is not None]
        if set_megatron:
            raise NotImplementedError(
                f'{set_megatron} are Megatron spellings that the transformers backend does not read (it uses '
                'learning_rate / max_steps / per_device_train_batch_size / ...), so a value here would be '
                'silently ignored. Drive Megatron through `swift.dev.cli.megatron` (which sets '
                'backend="megatron" and translates these), or pass the HF-named fields.')

    return (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
            logging_config, _select_tuner(tuner))


def sft_main(argv: Optional[List[str]] = None) -> List[dict]:
    from swift.dev.cli.runtime import bootstrap_run
    from swift.dev.config import process_configs, validate_configs
    from swift.dev.recipe import run_embedding, run_reranker, run_seq_cls, run_sft

    (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
     logging_config, tuner_config) = parse_sft_configs(argv)
    process_configs(
        model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
        tuner_config)
    validate_configs(
        model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
        tuner_config, logging_config=logging_config)
    bootstrap_run(model_config, checkpoint_config, dataset_config, tuner_config, seed=train_config.seed)

    task_type = model_config.task_type or 'causal_lm'
    recipes = {
        'causal_lm': run_sft,
        'seq_cls': run_seq_cls,
        'embedding': run_embedding,
        'reranker': run_reranker,
        'generative_reranker': run_reranker,
    }
    if task_type not in recipes:
        raise ValueError(f'Unsupported SFT task_type: {task_type!r}.')
    return recipes[task_type](
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        output_dir=checkpoint_config.output_dir,
    )


if __name__ == '__main__':
    sft_main()

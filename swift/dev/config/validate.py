# Cross-config validation: the one place where rules spanning several Configs are enforced.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.configs import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig, TunerConfig)

logger = logging.getLogger(__name__)


def validate_configs(
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    dataset_config: 'DatasetConfig',
    train_config: 'TrainConfig',
    distributed_config: 'DistributedConfig',
    checkpoint_config: Optional['CheckpointConfig'] = None,
    tuner_config: Optional['TunerConfig'] = None,
) -> None:
    """Validate constraints that span multiple Configs. Raises ValueError on an illegal combination.

    Call this BEFORE building anything heavy (dataset/model)
    """
    from swift.dev.builders.model import is_megatron_backend
    is_megatron = is_megatron_backend(distributed_config)

    _check_group_by_length(dataset_config, template_config)
    _check_lazy_tokenize(dataset_config)
    _check_packing(dataset_config, template_config)
    _check_data_sharding(dataset_config)
    _check_streaming(dataset_config, checkpoint_config)
    _check_backend_specific(dataset_config, train_config, distributed_config, is_megatron, tuner_config)
    _check_megatron_optimizer(train_config, is_megatron)
    _check_megatron_recompute(train_config, distributed_config, is_megatron)
    _check_megatron_attn_backend(model_config, template_config, is_megatron)


def _check_megatron_attn_backend(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                                 is_megatron: bool) -> None:
    """padding_free needs an attention kernel that supports variable-length (THD) input.

    legacy handles this by DOWNGRADING: on an 'unfused' backend it logs a warning and sets
    args.padding_free = False (swift/megatron/model/utils.py::_check_padding_free). dev raises
    instead, deliberately not mirroring that:
      - silently rewriting the user's request is the exact failure mode this refactor has been
        removing (the warmup rounding and the min_lr override both silently ran something else);
      - 'unfused' is never a default -- reaching it means the user typed both it and padding_free,
        i.e. asked for two things that cannot hold together, which is worth reporting;
      - the downgrade also silently changes throughput/memory, so a run could look fine and be much
        slower than the config implies.
    Recorded as a break-change rather than hidden.

    legacy's other attention guard (flash + softmax_type='learnable' -> raise) is NOT mirrored: dev
    has no softmax_type or experimental_attention_variant field, so mcore keeps its default
    ('vanilla') and the condition cannot be reached. Mirroring it would add a permanently-false
    branch. If either field is ever added to dev, that guard has to come with it.
    """
    if not is_megatron or not template_config.padding_free:
        return
    if model_config.attn_impl is None:
        return
    # Resolve first: 'sdpa' also lands on the unfused kernel, so comparing the raw string would miss
    # it. Unknown/unsupported values are not this guard's business -- resolve_megatron_attn_backend
    # reports those at build time with a better message.
    from megatron.core.transformer.enums import AttnBackend

    from swift.dev.naming import resolve_megatron_attn_backend
    try:
        backend = resolve_megatron_attn_backend(model_config.attn_impl)
    except NotImplementedError:
        return
    if backend is AttnBackend.unfused:
        raise ValueError(f'padding_free=True is incompatible with attn_impl={model_config.attn_impl!r} (the '
                         'unfused attention kernel): it does not support the variable-length (THD) layout '
                         'padding_free produces. legacy silently turns padding_free off here; dev refuses instead '
                         'so the run does not quietly train a different shape. Choose attn_impl="flash"/"fused", '
                         'or set padding_free=False.')


def _check_group_by_length(dataset_config: 'DatasetConfig', template_config: 'TemplateConfig') -> None:
    if not dataset_config.group_by_length:
        return
    if template_config.padding_free:
        raise ValueError('group_by_length is incompatible with padding_free: padding_free flattens each micro '
                         'batch into a single variable-length sequence, so there is no padding left for length '
                         'grouping to remove (and clustering long samples raises peak activation memory). '
                         'Set one of them to False.')
    if dataset_config.packing:
        raise ValueError('group_by_length is incompatible with packing: packing already bin-packs samples to '
                         '~packing_length and implies padding_free, so length grouping cannot help. '
                         'Use packing alone, or set group_by_length=False.')
    # Streaming has no random access and no precomputed `lengths` column, so the sampler cannot
    # group. Previously build_dataset passed group_by_length through WITHOUT lengths, so this died
    # later as an opaque 'lengths must be provided'; failing here reports the actual cause.
    if dataset_config.streaming:
        raise ValueError('group_by_length requires a map-style dataset: streaming datasets have no `lengths` '
                         'column and no random access, so samples cannot be reordered by length. '
                         'Set streaming=False or group_by_length=False.')
    # `lengths` only exists after an EAGER encode. lazy_tokenize=None means AUTO, and auto always
    # backs off to eager when group_by_length is on (see _encode_mode), so only an EXPLICIT opt-in
    # to lazy is a conflict here.
    if dataset_config.lazy_tokenize:
        raise ValueError('group_by_length requires lazy_tokenize=False: the per-sample `lengths` column it '
                         'sorts on is only produced by eager encoding (AddLengthPreprocessor).')


def _check_lazy_tokenize(dataset_config: 'DatasetConfig') -> None:
    """Explicit lazy_tokenize=True conflicts with packing / streaming (legacy base_args.py:136-140).

    Only the EXPLICIT opt-in is a conflict: None is auto, and auto backs off to eager whenever one
    of these is on, so it can never reach here in a violating state.
    """
    if not dataset_config.lazy_tokenize:
        return
    if dataset_config.packing:
        raise ValueError('packing and lazy_tokenize are incompatible: PackingDataset reads the '
                         '`lengths` column at construction (packing.py:78), which only eager '
                         'encoding writes.')
    if dataset_config.streaming:
        raise ValueError('streaming and lazy_tokenize are incompatible.')


def _check_packing(dataset_config: 'DatasetConfig', template_config: 'TemplateConfig') -> None:
    if not dataset_config.packing:
        return
    if not template_config.padding_free:
        logger.info('Setting padding_free True as packing is set')
        template_config.padding_free = True


def _check_data_sharding(dataset_config: 'DatasetConfig') -> None:
    """data_sharding needs a shuffled order to reshuffle; it is a no-op under sequential reads.

    The group_by_length conflict is intentionally NOT fatal here: legacy downgrades data_sharding to
    False with a warning (batch_sampler.py:86-90), and existing Megatron scripts that set both must
    keep running unchanged. build_dataset performs that downgrade.
    """
    if dataset_config.data_sharding and not dataset_config.train_dataloader_shuffle:
        raise ValueError('data_sharding requires train_dataloader_shuffle=True: it only changes the SCOPE of the '
                         'per-epoch reshuffle (shuffle within a rank shard vs. globally), so with shuffling off '
                         'it does nothing.')


def _check_streaming(dataset_config: 'DatasetConfig', checkpoint_config: Optional['CheckpointConfig']) -> None:
    """Streaming/iterable datasets cannot be resumed deterministically (no epoch-aware skip)."""
    # A cached_dataset is a map-style Dataset written by save_to_disk, so it cannot participate in
    # the streaming pipeline. Legacy asserts the same in SwiftSft._prepare_dataset
    # ('Cached dataset does not support streaming.').
    if dataset_config.streaming and (dataset_config.cached_dataset or dataset_config.cached_val_dataset):
        raise ValueError('cached_dataset does not support streaming=True: the exported cache is a map-style '
                         'dataset loaded via load_from_disk. Set streaming=False, or drop cached_dataset.')
    if checkpoint_config is None:
        return
    if dataset_config.streaming and checkpoint_config.resume_from_checkpoint:
        raise NotImplementedError('Resume is not supported for streaming/iterable datasets (no deterministic '
                                  'epoch-aware skip). Use a map-style dataset, or set resume_from_checkpoint=None.')


def _check_megatron_recompute(train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                              is_megatron: bool) -> None:
    """gradient_checkpointing decides nothing on Megatron; recompute_granularity does.

    build_model forwards only recompute_granularity/method/num_layers to MegatronModel, so the
    HF-named flag is unread there. It cannot go in the _HF_ONLY table: its default is True, so every
    existing Megatron run would start failing. The two directions differ in what they deserve --
    flag-on-but-nothing-configured is the DEFAULT state and can only warn, while flag-off-yet-
    recompute-configured is a contradiction the user typed and is fatal.

    The default also disagrees with legacy, which recomputes ('selective') unless told otherwise;
    aligning that would change dev's memory/throughput baseline, so it is recorded in the design doc
    rather than changed here.
    """
    if not is_megatron:
        return
    granularity = distributed_config.recompute_granularity
    if not train_config.gradient_checkpointing and granularity:
        raise ValueError(f'gradient_checkpointing=False contradicts recompute_granularity={granularity!r}: on the '
                         'Megatron backend recompute is driven by recompute_granularity alone, so the run WOULD '
                         'recompute. Drop one of the two.')
    if train_config.gradient_checkpointing and not granularity:
        logger.warning('gradient_checkpointing=True has no effect on the Megatron backend and this run will NOT '
                       'recompute: set DistributedConfig.recompute_granularity (legacy megatron defaults to '
                       "'selective') to enable it.")


def _check_megatron_optimizer(train_config: 'TrainConfig', is_megatron: bool) -> None:
    """Reject an inconsistent Megatron optimizer/scheduler config before the weights are loaded."""
    if not is_megatron:
        return
    from swift.dev.optimizer import megatron_weight_decay_bounds

    # Delegates so the weight-decay rule has one definition (configure_optimizer uses the same).
    megatron_weight_decay_bounds(train_config)
    # 'cosine_with_min_lr' is Megatron's plain cosine plus a floor, so without min_lr it would run as
    # ordinary cosine -- the name silently not doing what it says. (On the HF path transformers
    # itself raises when neither min_lr nor min_lr_rate is given.)
    if train_config.lr_scheduler_type.lower() == 'cosine_with_min_lr' and not train_config.min_lr:
        raise ValueError("lr_scheduler_type='cosine_with_min_lr' needs TrainConfig.min_lr > 0 on the Megatron "
                         'backend; with min_lr=0 it is just cosine. Set min_lr, or use '
                         "lr_scheduler_type='cosine'.")


def _is_off(value, off_value) -> bool:
    if off_value is None:
        # A falsy NUMBER is a real setting, a falsy container is not. start_weight_decay=0.0 (ramp
        # up from no decay) has to count as set or the wrong-backend check skips it, while the CLI
        # normalizes an unset fsdp to [] against a None default and must still count as unset.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return False
        return not value
    if not off_value:
        return not value
    return value == off_value


_MEGATRON_ONLY = (
    ('dataset_config', 'data_sharding', False),
    # NOTE: clip_grad is NOT here -- it is a deprecated alias of max_grad_norm and valid on both
    # backends (resolve_max_grad_norm folds the two). Listing it would reject legacy Megatron argv.
    ('train_config', 'weight_decay_incr_style', 'constant'),
    ('train_config', 'start_weight_decay', None),
    ('train_config', 'end_weight_decay', None),
    ('train_config', 'min_lr', 0.0),
    ('distributed_config', 'sequence_parallel', False),
    ('distributed_config', 'recompute_granularity', None),
    ('distributed_config', 'recompute_method', None),
    ('distributed_config', 'recompute_num_layers', None),
    ('distributed_config', 'tensor_model_parallel_size', 1),
    ('distributed_config', 'pipeline_model_parallel_size', 1),
    ('distributed_config', 'context_parallel_size', 1),
    ('distributed_config', 'expert_model_parallel_size', 1),
)

_HF_ONLY = (
    ('distributed_config', 'deepspeed', None),
    ('distributed_config', 'zero_hpz_partition_size', None),
    ('distributed_config', 'deepspeed_autotp_size', None),
    ('distributed_config', 'fsdp', None),
    ('distributed_config', 'ddp_find_unused_parameters', None),
    ('tuner_config', 'use_galore', False),
    ('tuner_config', 'lisa_activated_layers', 0),
    ('train_config', 'use_liger_kernel', False),
    ('train_config', 'neftune_noise_alpha', None),
    ('train_config', 'optim', 'adamw_torch_fused'),
    ('train_config', 'optim_args', None),
    ('train_config', 'gradient_checkpointing_kwargs', None),
    ('train_config', 'router_aux_loss_coef', 0.0),
    ('train_config', 'use_logits_to_keep', None),
    ('train_config', 'predict_with_generate', False),
    ('train_config', 'eval_use_evalscope', False),
    ('train_config', 'full_determinism', False),
)

_MEGATRON_PARALLEL_SIZES = (
    'tensor_model_parallel_size',
    'pipeline_model_parallel_size',
    'context_parallel_size',
    'expert_model_parallel_size',
)


def _check_backend_specific(dataset_config: 'DatasetConfig',
                            train_config: 'TrainConfig',
                            distributed_config: 'DistributedConfig',
                            is_megatron: bool,
                            tuner_config: Optional['TunerConfig'] = None) -> None:
    """Reject knobs the active backend does not implement, so they cannot be silently ignored."""
    holders = {
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'tuner_config': tuner_config,
    }
    offending = _MEGATRON_ONLY if not is_megatron else _HF_ONLY
    wrong_backend = 'transformers' if not is_megatron else 'megatron'
    right_backend = 'megatron' if not is_megatron else 'transformers'

    for holder_name, attr, off_value in offending:
        holder = holders[holder_name]
        # tuner_config is optional (None == full-param training), in which case its tuner-only
        # knobs cannot have been set at all -- nothing to check.
        if holder is None:
            continue
        value = getattr(holder, attr)
        if _is_off(value, off_value):
            continue
        hint = (f'the {wrong_backend} backend runs with all Megatron parallel sizes == 1'
                if attr in _MEGATRON_PARALLEL_SIZES else f'the active backend is {wrong_backend}')
        raise ValueError(f'{attr}={value!r} is only implemented by the {right_backend} backend, but {hint}. '
                         f'Remove it, or switch DistributedConfig.backend.')

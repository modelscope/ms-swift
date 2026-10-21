# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, get_args

from swift.dev.utils.logger import get_logger
from .rl_config import F_DIVERGENCE_TYPE

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        ConvertConfig,
        DatasetConfig,
        DistributedConfig,
        InferConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RewardConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()


def validate_configs(
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    dataset_config: 'DatasetConfig',
    train_config: 'TrainConfig',
    distributed_config: 'DistributedConfig',
    checkpoint_config: Optional['CheckpointConfig'] = None,
    tuner_config: Optional['TunerConfig'] = None,
    rl_config: Optional['RLConfig'] = None,
    reward_config: Optional['RewardConfig'] = None,
    rollout_config: Optional['RolloutConfig'] = None,
    logging_config: Optional['LoggingConfig'] = None,
    quantize_config: Optional['QuantizeConfig'] = None,
    megatron_config: Optional['MegatronConfig'] = None,
    moe_config: Optional['MoEConfig'] = None,
    *,
    training: bool = True,
) -> None:
    """Validate cross-Config constraints before anything heavy is built; raises ValueError."""
    from swift.dev.builders import is_megatron_backend
    is_megatron = is_megatron_backend(distributed_config)

    _check_group_by_length(dataset_config, template_config)
    _check_lazy_tokenize(dataset_config)
    _check_data_sharding(dataset_config)
    _check_streaming(dataset_config)
    _check_backend_specific(model_config, dataset_config, train_config, distributed_config, is_megatron, tuner_config)
    _check_galore(train_config, tuner_config)
    _check_param_groups(tuner_config)
    _check_unsloth_strategy(distributed_config, template_config)
    _check_deepspeed_autotp(distributed_config)
    _check_eval_generation(train_config, template_config, distributed_config)
    _check_megatron_runtime_configs(megatron_config, moe_config, is_megatron)
    _check_megatron_optimizer(train_config, is_megatron)
    _check_muon(train_config, distributed_config, is_megatron)
    _check_megatron_recompute(train_config, distributed_config, is_megatron)
    _check_megatron_microbatch_schedule(train_config, distributed_config, is_megatron)
    _check_eval_iters(train_config)
    _check_megatron_attn_backend(model_config, template_config, is_megatron)
    _check_padding_free_attn(model_config, template_config, dataset_config, is_megatron)
    _check_mtp(model_config, is_megatron, tuner_config)
    _check_quantization(model_config, distributed_config, is_megatron)
    _check_load_quantization(
        model_config, distributed_config, tuner_config, quantize_config, is_megatron, training=training)
    _check_megatron_fsdp(distributed_config, is_megatron)
    _check_selective_recompute(distributed_config, is_megatron)
    _check_pipeline_decoder_layers(distributed_config, is_megatron)
    _check_freeze_ratio_pp(train_config, distributed_config, is_megatron, tuner_config)
    _check_tp_comm_overlap(distributed_config, is_megatron)
    _check_sequence_parallel_tp(distributed_config, is_megatron)
    _check_checkpoint_runtime(
        checkpoint_config,
        distributed_config,
        tuner_config,
        rl_config,
        is_megatron,
        training=training)
    _check_save_total_limit(checkpoint_config, is_megatron)
    _check_logging(logging_config)
    _check_rlhf_ref_model(model_config, tuner_config, rl_config)
    _check_rlhf_advanced(train_config, rl_config, reward_config, rollout_config, distributed_config)
    _check_grpo_truncation_strategy(template_config, rl_config)
    _check_grpo_cached_dataset(dataset_config, rl_config)
    _check_rlhf_padding_free(template_config, dataset_config, rl_config)
    _check_rlhf_padding_side(template_config, rl_config)
    _check_rlhf_sequence_parallel(
        model_config, template_config, dataset_config, distributed_config, rl_config, is_megatron)
    # Packing-derived padding_free has already been resolved by process_configs. RLHF SP is fully governed
    # by _check_rlhf_sequence_parallel above, so this early-returns for a non-None rl_config.
    _check_hf_sequence_parallel(
        model_config, template_config, dataset_config, distributed_config, is_megatron, rl_config)


def _changed_fields(config) -> list:
    """User-visible fields that differ from defaults or were explicit on the CLI."""
    import dataclasses

    if config is None:
        return []
    explicit = getattr(config, '_explicit_fields', None)
    if explicit is not None:
        return sorted(explicit)
    changed = []
    for config_field in dataclasses.fields(config):
        if config_field.default is not dataclasses.MISSING:
            default = config_field.default
        elif config_field.default_factory is not dataclasses.MISSING:
            default = config_field.default_factory()
        else:
            continue
        if getattr(config, config_field.name) != default:
            changed.append(config_field.name)
    return changed


def _check_megatron_runtime_configs(megatron_config: Optional['MegatronConfig'], moe_config: Optional['MoEConfig'],
                                     is_megatron: bool) -> None:
    if not is_megatron:
        changed = _changed_fields(megatron_config) + _changed_fields(moe_config)
        if changed:
            raise ValueError(f'Megatron-only options {changed} cannot be used with the transformers backend.')
        return
    if megatron_config is None:
        return
    unsupported = {
        'apply_dsa_kernel_fusion',
        'csa_dense_mode',
        'sequence_packing_scheduler',
        'use_fused_mhc',
    }
    requested = sorted(unsupported.intersection(_changed_fields(megatron_config)))
    if requested:
        raise NotImplementedError(
            f'{requested} are not exposed by the installed mcore-bridge ModelConfig. '
            'Upgrade mcore-bridge/Megatron-LM or remove these options.')
    if megatron_config.manual_gc_steps < 0:
        raise ValueError('manual_gc_steps must be >= 0.')


def _check_rlhf_advanced(train_config: 'TrainConfig', rl_config: Optional['RLConfig'],
                         reward_config: Optional['RewardConfig'] = None,
                         rollout_config: Optional['RolloutConfig'] = None,
                         distributed_config: Optional['DistributedConfig'] = None) -> None:
    """Validate online-RL features before models, teachers, or rollout workers are created.

    Reward-channel surface on reward_config, multi-turn on rollout_config; the multi-turn GRPO-only
    gate lives in _check_rollout_tools.
    """
    if rl_config is None:
        return
    cfg = rl_config
    online_only = bool(cfg.chord_sft_dataset or cfg.advantage_reweight or cfg.sdar_loss_coef > 0
                       or cfg.dynamic_sample)
    if online_only and cfg.rl_type != 'grpo':
        raise ValueError('CHORD, RLSD, SDAR, and dynamic sampling are GRPO-only.')
    if cfg.sync_ref_model and cfg.rl_type not in ('grpo', 'ppo'):
        raise ValueError('sync_ref_model requires rl_type grpo or ppo (only they have a reference model).')
    _check_dynamic_sampling(cfg, reward_config)
    _check_grpo_controls(cfg)
    _check_reference_sync(cfg, train_config, distributed_config)
    _check_chord(cfg)
    _check_router_replay(cfg)
    _check_sampling_replay(cfg, rollout_config)
    _check_prm(cfg, reward_config)
    _check_self_distillation(cfg, train_config, reward_config)
    _check_distillation(cfg)
    _check_auxiliary_adapters(cfg, reward_config)
    _check_preference_reference(cfg)
    _check_dpo_loss(cfg)


# The f-divergences DPO can stand in for the KL; derived from F_DIVERGENCE_TYPE so the runtime check and
# the config field type share one source of truth and cannot drift.
_DPO_F_DIVERGENCE_TYPES = get_args(F_DIVERGENCE_TYPE)


def _check_dpo_loss(cfg: 'RLConfig') -> None:
    """Validate the DPO f-divergence name and the multi-loss (MPO) weight alignment (dpo-only knobs)."""
    if cfg.rl_type != 'dpo':
        return
    if cfg.f_divergence_type not in _DPO_F_DIVERGENCE_TYPES:
        raise ValueError(f'--f_divergence_type must be one of {list(_DPO_F_DIVERGENCE_TYPES)}, '
                         f'got {cfg.f_divergence_type!r}.')
    loss_types = cfg.loss_type or ['sigmoid']
    if cfg.loss_weights is not None and len(cfg.loss_weights) != len(loss_types):
        raise ValueError(f'--loss_weights must give one weight per --loss_type '
                         f'({len(cfg.loss_weights)} weights for {len(loss_types)} loss types).')


def _check_preference_reference(cfg: 'RLConfig') -> None:
    """Reject reference-free KTO: its implicit reward AND z_KL anchor are both log-ratios vs the reference."""
    if cfg.rl_type == 'kto' and cfg.reference_free:
        raise ValueError('KTO cannot be reference-free: its implicit reward is the log-ratio against a '
                         'reference and z_KL is a KL against that reference. Remove --reference_free; '
                         'under LoRA the adapter-disabled base serves as the reference automatically, and '
                         'full fine-tuning derives one from the policy init.')


def validate_multi_turn_config(rollout_config: Optional['RolloutConfig']) -> None:
    """Validate the shared multi-turn surface without RL-training-only constraints."""
    _check_multi_turn(rollout_config)


def _check_multi_turn(rollout_config: Optional['RolloutConfig']) -> None:
    # Multi-turn is on iff max_turns is set; per-turn length is sampling_params.max_tokens,
    # max_trajectory_tokens caps the whole trajectory. Both are independent, either may be None.
    if rollout_config is None or rollout_config.max_turns is None:
        return
    if rollout_config.max_turns < 1:
        raise ValueError('max_turns must be >= 1.')
    if rollout_config.max_trajectory_tokens is not None and rollout_config.max_trajectory_tokens < 1:
        raise ValueError('max_trajectory_tokens must be >= 1.')


def validate_rollout_config(rollout_config: Optional['RolloutConfig'],
                            rl_config: Optional['RLConfig'] = None,
                            tuner_config: Optional['TunerConfig'] = None) -> None:
    """Validate the rollout tools/sandbox surface without RL-training-only constraints.

    max_turns lives on the rollout_config itself; its GRPO-only gate runs here too
    (_check_rollout_tools). rl_config/tuner_config are read only by _check_rollout_tools and
    _check_async_mode (no-op at async_mode='none').
    """
    _check_rollout_sampler(rollout_config)
    _check_rollout_tools(rollout_config, rl_config)
    _check_offload_knobs(rollout_config)
    _check_resource_mode(rollout_config, rl_config)
    _check_async_mode(rollout_config, rl_config, tuner_config)


def _check_resource_mode(rollout_config: Optional['RolloutConfig'], rl_config: Optional['RLConfig']) -> None:
    """Validate static/adaptive resource planning before any Ray actor is created."""
    if rollout_config is None:
        return
    mode = rollout_config.resource_mode
    if mode not in {'manual', 'estimate', 'adaptive'}:
        raise ValueError(f'unknown resource_mode {mode!r}; expected manual, estimate, or adaptive.')
    resource_fields = {
        'resource_total_gpus', 'resource_gpu_memory_gib', 'resource_flex_gpus', 'resource_window_steps',
        'resource_min_hold_steps', 'resource_overrides'
    }
    changed = _changed_fields(rollout_config)
    if mode == 'manual':
        # _changed_fields returns a list; intersection accepts any iterable while ``&`` needs a set.
        inert = sorted(resource_fields.intersection(changed))
        if inert:
            raise ValueError(f'resource_mode="manual" does not read {inert}; select resource_mode="estimate" '
                             'or "adaptive", or remove these resource settings.')
        return
    if rl_config is None or rl_config.rl_type not in {'grpo', 'ppo', 'gkd', 'opsd', 'mopd'}:
        raise ValueError(f'resource_mode={mode!r} is implemented only by grpo/ppo/gkd/opsd/mopd online recipes.')
    if rollout_config.resource_total_gpus is not None and rollout_config.resource_total_gpus < 1:
        raise ValueError('resource_total_gpus must be >= 1 when set.')
    if rollout_config.resource_gpu_memory_gib is not None and rollout_config.resource_gpu_memory_gib <= 0:
        raise ValueError('resource_gpu_memory_gib must be > 0 when set.')
    if rollout_config.resource_flex_gpus < 0:
        raise ValueError('resource_flex_gpus must be >= 0.')
    if rollout_config.resource_window_steps < 1:
        raise ValueError('resource_window_steps must be >= 1.')
    if rollout_config.resource_min_hold_steps < 1:
        raise ValueError('resource_min_hold_steps must be >= 1.')
    if rollout_config.resource_overrides is not None and not isinstance(rollout_config.resource_overrides, dict):
        raise ValueError('resource_overrides must be a JSON object.')
    if mode != 'adaptive':
        return
    if rollout_config.sampler_mode != 'disaggregated':
        raise ValueError('resource_mode="adaptive" keeps dedicated sampler actors and optionally borrows the '
                         'complete trainer world; set sampler_mode="disaggregated" instead of configuring the '
                         'legacy exclusive colocate path.')
    if rollout_config.max_turns is not None:
        raise ValueError('resource_mode="adaptive" does not support multi-turn rollout in the first version; '
                         'use resource_mode="estimate" or disable max_turns.')
    if rl_config.rl_type in {'gkd', 'opsd', 'mopd'} and rl_config.lmbda != 1.0:
        raise ValueError('resource_mode="adaptive" requires lmbda=1 for distillation recipes: off-policy '
                         'dataset rounds use the synchronous round loop and have no streaming publish boundary '
                         'where the adaptive controller can move a complete replica. Set lmbda=1 or use '
                         'resource_mode="estimate".')
    if rollout_config.weight_sync_strategy != 'in_place' or not rollout_config.allow_partial_rollout:
        raise ValueError('resource_mode="adaptive" requires weight_sync_strategy="in_place" and '
                         'allow_partial_rollout=true so shared generations can pause across handovers.')
    if rollout_config.sleep_level not in (0, 1):
        raise ValueError('resource_mode="adaptive" supports only sleep_level 0 or 1; level 2 discards weights '
                         'needed by the shared sampler handover.')


# Online recipes that roll out per prompt batch and train on the collected samples (overlap generation
# with training on the shared streaming driver). RFT and the offline RLHF family are excluded.
_ASYNC_RLHF_TYPES = frozenset({'grpo', 'ppo', 'gkd', 'opsd', 'mopd'})


# Streaming knobs read ONLY by the per-sample streaming driver: all four under an OVERLAPPING
# async_mode; staleness-0 streaming all but max_staleness/allow_partial_rollout; sync-only none.
_FULLY_ASYNC_ONLY_DEFAULTS = {
    'max_staleness': 1,
    'weight_sync_strategy': 'adapter_snapshot',
    'allow_partial_rollout': False,
    'parameter_sync_step': 1
}

# The subset inert even when async_mode='none' rides the streaming driver: the dispatch pins max_staleness
# to 0 and the staleness-0 drain makes allow_partial_rollout a no-op, so neither is read on that path.
_INERT_UNDER_NONE_STREAMING = {
    'max_staleness': _FULLY_ASYNC_ONLY_DEFAULTS['max_staleness'],
    'allow_partial_rollout': _FULLY_ASYNC_ONLY_DEFAULTS['allow_partial_rollout'],
}


def _sync_only_feature(rl_config: Optional['RLConfig']) -> bool:
    """Whether this config runs the synchronous round loop instead of the per-sample stream.

    Mirrors the run_* dispatch: only distillation with lmbda != 1.0 and a None rl_config are sync-only.
    """
    if rl_config is None:
        return True
    if rl_config.rl_type in {'gkd', 'opsd', 'mopd'}:
        return rl_config.lmbda != 1.0
    return False


def _streams_under_none(rl_config: Optional['RLConfig']) -> bool:
    """Whether async_mode='none' routes this config to the streaming driver (staleness 0).

    True for an online loop with no sync-only feature; False for the off-policy mix and non-online.
    """
    if rl_config is None or rl_config.rl_type not in _ASYNC_RLHF_TYPES:
        return False
    return not _sync_only_feature(rl_config)


def _check_async_mode(rollout_config: Optional['RolloutConfig'], rl_config: Optional['RLConfig'],
                      tuner_config: Optional['TunerConfig'] = None) -> None:
    """Reject async_mode combinations the online loops cannot honour (fail-loudly).

    Refused, not degraded: non-online recipes, no IS correction, colocate, remax, off-policy mix.
    """
    if rollout_config is None:
        return
    # allow_partial_rollout is only MEANINGFUL under an OVERLAPPING streaming publish; at staleness 0 the
    # drain empties the in-flight set, so it is inert there (refused below if set explicitly).
    mode = rollout_config.async_mode
    if mode == 'none':
        # An online loop with no sync-only feature rides the StreamingDriver at staleness 0; only the
        # distillation off-policy mix (lmbda != 1.0) runs its synchronous round loop instead.
        if _streams_under_none(rl_config):
            # max_staleness (pinned to 0) and allow_partial_rollout (inert at staleness 0) are never read on
            # this path; weight_sync_strategy and parameter_sync_step ARE, validated by the publication check.
            inert = _INERT_UNDER_NONE_STREAMING
            if rollout_config.resource_mode == 'adaptive':
                # Adaptive shared handover may pause a long request at every batch boundary even at
                # staleness 0, so partial rollout is active and required there.
                inert = {'max_staleness': _INERT_UNDER_NONE_STREAMING['max_staleness']}
            _reject_inert_streaming_knobs(rollout_config, mode, inert)
            _check_streaming_publication(rollout_config, rl_config, tuner_config, mode)
        else:
            # The synchronous round loop (or a non-online recipe) reads none of the streaming knobs.
            _reject_inert_streaming_knobs(rollout_config, mode, _FULLY_ASYNC_ONLY_DEFAULTS)
        return
    # The legacy bool is the old spelling of 'one_step_off'; process folds it in when async_mode is
    # unset, so reaching here with async_generate True AND a different explicit mode is ambiguous.
    if rollout_config.async_generate and mode != 'one_step_off':
        raise ValueError(
            f'--async_generate is the legacy spelling of --async_mode one_step_off, but --async_mode '
            f'{mode!r} was also set. Drop --async_generate and express the regime with --async_mode alone.')
    flag = '--async_generate' if rollout_config.async_generate else f'--async_mode {mode}'
    rl_type = None if rl_config is None else rl_config.rl_type
    if rl_type not in _ASYNC_RLHF_TYPES:
        raise ValueError(
            f'{flag} is only implemented for the online loops that roll out per prompt batch and '
            f'train on the collected samples (grpo/ppo/gkd/opsd/mopd); got rl_type={rl_type!r}. The '
            'other recipes override the training loop (or never roll out) and would silently ignore it. '
            'Remove the async setting.')
    if rl_type == 'grpo' and rl_config.rollout_importance_sampling_mode is None:
        raise ValueError(
            f'{flag} trains GRPO on a rollout batch that is stale (off-policy), so it requires off-policy '
            'correction: set --rollout_importance_sampling_mode (e.g. token_truncate or sequence_mask). '
            'Use --async_mode none if you want a strictly on-policy loop.')
    if rollout_config.sampler_mode != 'disaggregated':
        raise ValueError(
            f'{flag} needs the sampler on its own GPUs to overlap generation with training, but '
            f"sampler_mode={rollout_config.sampler_mode!r}. Set --sampler_mode disaggregated; 'colocate' time-shares "
            'one DeviceGroup and cannot overlap.')
    if rl_config.dynamic_sample:
        raise ValueError(
            f'{flag} cannot be combined with --dynamic_sample: dynamic sampling regenerates '
            'zero-variance groups after rewards are seen, which cannot be pre-submitted ahead of training. '
            'Disable one of them.')
    if rl_config.advantage_estimator == 'remax':
        raise ValueError(
            f'{flag} cannot be combined with --advantage_estimator remax: ReMax scores a greedy '
            '(argmax) baseline completion per prompt AFTER the sampled rollout, and that extra synchronous '
            'generation pass cannot be pre-submitted ahead. Remove the async setting or pick another '
            'advantage_estimator.')
    if rl_type in {'gkd', 'opsd', 'mopd'} and rl_config.lmbda != 1.0:
        raise ValueError(
            f'{flag} overlaps generation with training by pre-submitting the next prompt batch, '
            f'which only purely on-policy distillation can do; rl_type={rl_type!r} with '
            f'--lmbda={rl_config.lmbda} mixes in off-policy dataset rounds that generate nothing and '
            'cannot be pre-submitted. Set --lmbda 1 for on-policy distillation, or remove the async setting.')
    if mode == 'fully_async' and rl_type not in ('grpo', 'ppo'):
        # The distillation family overlaps a SINGLE step (staleness 1) on the same per-sample streaming
        # driver and has no deep-buffer regime, so fully_async (staleness > 1) stays GRPO/PPO-only.
        raise ValueError(
            f'{flag} is only implemented for GRPO and PPO; got rl_type={rl_type!r}. The '
            'distillation loops overlap a single step -- use --async_mode one_step_off for them.')
    # Every surviving overlapping regime runs the SAME per-sample StreamingDriver (differing only in the
    # staleness knob), so all validate their publication identically below.
    _check_streaming_publication(rollout_config, rl_config, tuner_config, mode)


def _reject_inert_streaming_knobs(rollout_config: 'RolloutConfig', mode: str, inert_defaults: dict) -> None:
    """Refuse a streaming knob explicitly set to a value the selected driver never reads.

    inert_defaults maps each inert knob to its forced value (full set on the sync fallback).
    """
    changed = _changed_fields(rollout_config)
    for name, inert in inert_defaults.items():
        if name in changed and getattr(rollout_config, name) != inert:
            raise ValueError(
                f'--{name}={getattr(rollout_config, name)!r} is inert on the driver this config selects '
                f'(--async_mode {mode}): its value there is always {inert!r}, so setting it would silently do '
                'nothing. Drop it, or use a regime that reads it.')


def _check_streaming_publication(rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
                                 tuner_config: Optional['TunerConfig'], mode: str) -> None:
    """Publication guards for the per-sample streaming driver (every online regime that rides it).

    Checks sync_step>=1, staleness per regime, colocate->in_place, and the strategy/partial pairing.
    """
    if rollout_config.parameter_sync_step < 1:
        raise ValueError(
            f'--parameter_sync_step must be >= 1 (one weight publication every K recorded steps); got '
            f'{rollout_config.parameter_sync_step}. 0 would never publish a trained policy to the sampler.')
    if mode == 'one_step_off':
        changed = _changed_fields(rollout_config)
        if 'max_staleness' in changed and rollout_config.max_staleness != 1:
            raise ValueError(
                f'--max_staleness={rollout_config.max_staleness!r} cannot be set under --async_mode '
                'one_step_off: that regime pins the streaming driver to a single lookahead window '
                '(staleness 1). Use --async_mode fully_async with --max_staleness N for a deeper buffer, '
                'or drop --max_staleness.')
    elif mode == 'fully_async' and rollout_config.max_staleness < 1:
        raise ValueError(
            f'--max_staleness must be >= 1 under --async_mode fully_async (0 would admit nothing ahead of '
            f'the oldest untrained window, i.e. the synchronous loop); got {rollout_config.max_staleness}.')
    strategy = rollout_config.weight_sync_strategy
    tuner = getattr(tuner_config, 'tuner', 'full') if tuner_config is not None else 'full'
    if rollout_config.sampler_mode == 'colocate' and strategy != 'in_place':
        raise ValueError(
            f"--sampler_mode colocate with --weight_sync_strategy {strategy!r} cannot hand the shared device "
            "back to the sampler each cycle: a colocated sampler time-shares ONE DeviceGroup with the trainer, "
            "and the per-cycle wake+sync that returns it to generation rides in_place's publish, whereas "
            "adapter_snapshot publishes by writing a new adapter path with no weight sync. Use "
            "--weight_sync_strategy in_place (what dev derives for a colocated run); a colocated run is "
            "synchronous (--async_mode none), so in_place needs no --allow_partial_rollout there.")
    if strategy == 'adapter_snapshot':
        # Per-version LoRA pinning needs an adapter to pin; a full-parameter policy has none. The sound
        # full-parameter stream is in_place (below), so point the user there.
        if tuner == 'full':
            raise ValueError(
                f"--async_mode {mode} with --weight_sync_strategy adapter_snapshot pins each trained "
                "version as a LoRA adapter path, but this is a full-parameter run (tuner='full') with no "
                'adapter to pin. Configure an adapter-based tuner (e.g. LoRA), or use --weight_sync_strategy '
                'in_place, which overwrites the sampler\'s live merged base weights (add '
                '--allow_partial_rollout for an overlapping regime, which resumes each in-flight generation).')
        if rollout_config.allow_partial_rollout:
            raise ValueError(
                '--allow_partial_rollout is inert under --weight_sync_strategy adapter_snapshot: each version '
                'is pinned as its own adapter path, so publishing never overwrites a weight copy an in-flight '
                'generation decodes and there is nothing to interrupt and resume. Drop --allow_partial_rollout, '
                'or use --weight_sync_strategy in_place (which overwrites one live copy and so needs it).')
    elif strategy == 'in_place':
        # Overwriting the live copy under an in-flight generation is unsound unless each is aborted and
        # resumed on fresh weights (allow_partial_rollout); the staleness-0 drain makes it a no-op there.
        if mode != 'none' and not rollout_config.allow_partial_rollout:
            raise ValueError(
                f"--async_mode {mode} with --weight_sync_strategy in_place overwrites the sampler's single "
                'live weight copy while the streaming buffer still has generations in flight, so it needs '
                '--allow_partial_rollout: each publish then aborts every in-flight generation and resumes it '
                'from its own tokens on the fresh weights (twinkle PartialRolloutMixin + abort-on-publish). '
                'Without it the rest of an in-flight generation would decode under half-updated weights whose '
                'logprobs match no consistent policy, which importance sampling cannot correct. Set '
                '--allow_partial_rollout, or use --weight_sync_strategy adapter_snapshot (per-version LoRA '
                'pinning, no interrupt).')
    else:
        raise ValueError(
            f"--weight_sync_strategy must be 'adapter_snapshot' or 'in_place' under --async_mode {mode}; "
            f'got {strategy!r}.')


def _check_offload_knobs(rollout_config: Optional['RolloutConfig']) -> None:
    """Warn on offload/sleep knobs dev cannot honour exactly as set (non-fatal).

    Colocate offload is atomic; disaggregated has its own GPUs, so offload is moot and sleep inert.
    """
    if rollout_config is None:
        return
    changed = _changed_fields(rollout_config)
    model_set = 'offload_model' in changed
    optim_set = 'offload_optimizer' in changed
    offload_model = rollout_config.offload_model
    offload_optimizer = rollout_config.offload_optimizer
    if rollout_config.sampler_mode == 'colocate':
        if model_set and optim_set and not offload_model and not offload_optimizer:
            logger.warning(
                'colocate rollout offload is structural and atomic (twinkle offload_to_cpu bundles the '
                'model and optimizer and is coupled to the wake/sleep handover), so it cannot be disabled: '
                'ignoring --offload_model=false --offload_optimizer=false; both stay offloaded each step.')
        elif (model_set or optim_set) and not (model_set and optim_set and offload_model and offload_optimizer):
            logger.warning(
                'colocate rollout offload is atomic: the model and optimizer are offloaded together and '
                'cannot be controlled as two independent tiers. Ignoring the partial --offload_model/'
                '--offload_optimizer setting; both stay offloaded each step.')
    else:
        if (model_set and offload_model) or (optim_set and offload_optimizer):
            logger.warning(
                'sampler_mode=%r places the sampler on its own GPUs, so offloading the trainer model/optimizer '
                'to make room for it does not apply: ignoring --offload_model/--offload_optimizer.',
                rollout_config.sampler_mode)
        if 'sleep_level' in changed:
            logger.warning(
                'sampler_mode=%r does not share GPUs between trainer and sampler, so the sampler is never put '
                'to sleep between steps: --sleep_level is inert here (it only takes effect under '
                "--sampler_mode colocate).", rollout_config.sampler_mode)


def _check_rollout_sampler(rollout_config: Optional['RolloutConfig']) -> None:
    """Reject a rollout engine dev cannot wire as a weight-synced sampler (vllm/sglang only).

    Weight sync needs a CheckpointEngineMixin engine; TransformersSampler has none by design.
    """
    if rollout_config is None:
        return
    if rollout_config.rollout_sampler not in ('vllm', 'sglang'):
        raise ValueError(
            f"rollout_sampler={rollout_config.rollout_sampler!r} cannot back an online-RL rollout: weight "
            "sync into the sampler needs a CheckpointEngineMixin engine, which only 'vllm' and 'sglang' "
            "provide (TransformersSampler has none by design). Use rollout_sampler='vllm' or 'sglang'.")


def _check_rollout_tools(rollout_config: Optional['RolloutConfig'],
                         rl_config: Optional['RLConfig'] = None) -> None:
    if rollout_config is None:
        return
    if rollout_config.sandbox_num_envs < 1:
        raise ValueError(f'sandbox_num_envs must be >= 1, got {rollout_config.sandbox_num_envs}.')
    # Multi-turn is orchestrated by the GRPO recipe only; sampling (rl_config is None, the TUI) is exempt.
    if rollout_config.max_turns is not None and rl_config is not None and rl_config.rl_type != 'grpo':
        raise ValueError(f'multi-turn rollout (max_turns) is GRPO-only, but rl_type={rl_config.rl_type!r}: the '
                         'GRPO recipe is the only one that assembles multi-turn trajectories. Use '
                         'rl_type=grpo, or leave max_turns unset.')
    if not rollout_config.tools:
        return
    # Tools are injected per-turn by the multi-turn engine, so a single-turn rollout has no turn to
    # call them in -- requiring max_turns turns a silent no-op into a clear error.
    if rollout_config.max_turns is None:
        raise ValueError(
            'RolloutConfig.tools requires a multi-turn rollout: set max_turns (>= 1) so the engine has '
            'turns in which to call the tools. A single-turn rollout cannot invoke them.')


def validate_infer_config(infer_config: Optional['InferConfig']) -> None:
    """Validate the best-of-n synthesis surface without RL-training-only constraints."""
    if infer_config is None:
        return
    from swift.dev.dataset.storage import STORAGE_MAPPING, validate_store_format
    # A store format is an open registry key, so check the name resolves (and is writable) here -- after
    # --external_plugins is imported -- rather than after the run has generated rows it cannot write.
    validate_store_format(infer_config.store_format, writable=True)
    # Resume eligibility is the format's own incremental capability, not a hard-coded name -- a plugin
    # declaring incremental=True earns it the same way. Non-incremental formats write once at the end.
    if not STORAGE_MAPPING[infer_config.store_format].incremental and infer_config.resume:
        raise ValueError(f"store_format={infer_config.store_format!r} writes the whole run once at the end "
                         'and cannot checkpoint, so it cannot be combined with resume=True. Use an '
                         'incremental store_format (jsonl) for a resumable run.')


def validate_convert_config(convert_config: Optional['ConvertConfig']) -> None:
    """Validate the export-side store choice, but only where a store is actually written.

    store_format is consumed solely by --to_cached_dataset; runs after --external_plugins is imported.
    """
    if convert_config is None or not convert_config.to_cached_dataset:
        return
    from swift.dev.dataset.storage import validate_store_format
    validate_store_format(convert_config.store_format, writable=True)


# The GRPO-family loss_type values dev accepts (each maps onto a twinkle GRPOLoss subclass). One constant
# so the GRPO-only gate in _check_grpo_controls and the support check in _check_grpo_loss_type cannot drift.
_GRPO_LOSS_TYPES = frozenset({
    'grpo', 'dapo', 'fipo', 'gspo', 'sapo', 'cispo', 'bnpo', 'dr_grpo', 'real', 'gpg', 'dro', 'dppo_tv',
    'dppo_kl', 'geo_mean', 'clip_cov', 'kl_cov'
})


def _check_grpo_controls(cfg: 'RLConfig') -> None:
    # cfg.loss_type is Optional[List[str]]; a GRPO run names at most one variant (enforced in
    # _check_grpo_loss_type) and 'grpo' is the resolver's fallback when it is unset (loss/configure.py).
    loss_type = cfg.loss_type[0] if cfg.loss_type else 'grpo'
    grpo_only = bool(
        cfg.log_completions or cfg.num_iterations != 1 or cfg.delta is not None
        or cfg.importance_sampling_level != 'token' or cfg.overlong_filter or cfg.log_entropy
        or cfg.top_entropy_quantile != 1.0 or cfg.rollout_importance_sampling_mode
        or cfg.log_rollout_offpolicy_metrics or cfg.off_policy_sequence_mask_delta is not None
        # A GRPO-SPECIFIC variant named on a non-GRPO run; the plain 'grpo' default is excluded on
        # purpose (the fallback for an UNSET loss_type -- counting it would flag every dpo/kto run).
        or (loss_type in _GRPO_LOSS_TYPES and loss_type != 'grpo'))
    if cfg.rl_type != 'grpo':
        if grpo_only:
            raise ValueError('GRPO clipping, replay, entropy, FIPO, and completion logging controls are GRPO-only.')
        return
    _check_grpo_loss_type(cfg, loss_type)
    # GRPO's advantage is group-relative: one generation per prompt has no baseline and degenerates to
    # raw-reward REINFORCE (legacy rejects it too). A teacher (OPD-RL) is exempt.
    has_teacher = bool(cfg.teacher_model or cfg.teacher_adapters or cfg._teacher_use_disable_adapter)
    if cfg.num_generations < 2 and not has_teacher:
        raise ValueError(f'GRPO requires num_generations >= 2 (got {cfg.num_generations}): its advantage is '
                         'group-relative, so one completion per prompt has no group baseline and degenerates '
                         'to raw-reward REINFORCE. Raise --num_generations, or use a teacher (OPD-RL) / '
                         '--rl_type rft for a single-sample objective.')
    positive = {
        'num_iterations': cfg.num_iterations,
        'rollout_importance_sampling_threshold': cfg.rollout_importance_sampling_threshold,
        'fipo_decay_rate': cfg.fipo_decay_rate,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f'GRPO controls must be > 0: {invalid}.')
    optional_nonnegative = {
        'delta': cfg.delta,
        'off_policy_sequence_mask_delta': cfg.off_policy_sequence_mask_delta,
        'fipo_clip_range': cfg.fipo_clip_range,
        'fipo_safety_threshold': cfg.fipo_safety_threshold,
    }
    invalid = [name for name, value in optional_nonnegative.items() if value is not None and value < 0]
    if invalid:
        raise ValueError(f'GRPO controls must be >= 0 when set: {invalid}.')
    if not 0.0 < cfg.top_entropy_quantile <= 1.0:
        raise ValueError('top_entropy_quantile must be in (0, 1].')


def _check_grpo_loss_type(cfg: 'RLConfig', loss_type: str) -> None:
    supported = _GRPO_LOSS_TYPES
    if len(cfg.loss_type or []) > 1:
        raise ValueError('GRPO supports exactly one loss_type.')
    if loss_type not in supported:
        raise ValueError(f'Unsupported GRPO loss_type={loss_type!r}; expected one of {sorted(supported)}.')
    # GSPO IS sequence-level importance sampling; an explicitly-set token-level request contradicts it
    # (the default 'token' is promoted to 'sequence', so only the explicit one is a real contradiction).
    if (loss_type == 'gspo' and cfg.importance_sampling_level == 'token'
            and 'importance_sampling_level' in _changed_fields(cfg)):
        raise ValueError(
            'loss_type=gspo IS sequence-level importance sampling; --importance_sampling_level token contradicts '
            'it and would silently downgrade GSPO to token-level GRPO. Drop the flag (gspo defaults to '
            'sequence-level), or use loss_type=grpo for token-level IS.')
    _check_grpo_loss_knobs(cfg, loss_type)


def _check_grpo_loss_knobs(cfg: 'RLConfig', loss_type: str) -> None:
    """Fail-loudly on the verl-ported loss variants' hyperparameters before training starts.

    The twinkle constructors re-validate; checking here surfaces a bad value in the argument pass.
    """
    if loss_type == 'dro' and cfg.dro_beta <= 0:
        raise ValueError(f'loss_type=dro requires dro_beta > 0, got {cfg.dro_beta}.')
    if loss_type in ('dppo_tv', 'dppo_kl') and cfg.clip_ratio_c <= 0:
        raise ValueError(f'loss_type={loss_type} requires clip_ratio_c > 0, got {cfg.clip_ratio_c}.')
    if loss_type == 'clip_cov' and cfg.clip_cov_ratio <= 0:
        raise ValueError(f'loss_type=clip_cov requires clip_cov_ratio > 0, got {cfg.clip_cov_ratio}.')
    if loss_type == 'clip_cov' and not cfg.clip_cov_lb < cfg.clip_cov_ub:
        raise ValueError(f'loss_type=clip_cov requires clip_cov_lb < clip_cov_ub, got '
                         f'{cfg.clip_cov_lb} >= {cfg.clip_cov_ub}.')
    if loss_type == 'kl_cov' and cfg.kl_cov_ratio <= 0:
        raise ValueError(f'loss_type=kl_cov requires kl_cov_ratio > 0, got {cfg.kl_cov_ratio}.')


def _check_grpo_cached_dataset(dataset_config: 'DatasetConfig', rl_config: Optional['RLConfig']) -> None:
    """Reject a pre-encoded cached_dataset for GRPO, which rolls out from raw prompt messages.

    Its prompt loader reads DatasetConfig.dataset only, so a cached_dataset would be silently ignored.
    """
    if rl_config is None or getattr(rl_config, 'rl_type', None) != 'grpo':
        return
    if dataset_config.cached_dataset or dataset_config.cached_val_dataset:
        raise ValueError('cached_dataset is not supported for GRPO: GRPO rolls out completions from each '
                         "prompt's raw `messages`, and its prompt loader reads DatasetConfig.dataset only, "
                         'so a cached_dataset (pre-encoded input_ids/labels, no messages) would be silently '
                         'ignored. Pass the original prompts via --dataset, or pre-generate completions '
                         'offline and train with an offline objective (dpo/rm/...).')


def _check_grpo_truncation_strategy(template_config: 'TemplateConfig', rl_config: Optional['RLConfig']) -> None:
    """GRPO truncates over-long prompts from the left or drops them; never 'right'/'split'.

    'right' cuts the prompt end the completion is conditioned on; 'split' breaks one prompt into two.
    """
    if rl_config is None or getattr(rl_config, 'rl_type', None) != 'grpo':
        return
    strategy = template_config.truncation_strategy
    if strategy in (None, 'left', 'delete'):
        return
    raise ValueError(f"GRPO requires truncation_strategy 'left' or 'delete' (or None), got {strategy!r}: "
                      "'right' cuts the end of the prompt -- the part the completion is conditioned on -- "
                      "and 'split' would split one prompt row into two rollout prompts. "
                      'Set --truncation_strategy left or delete.')


def _check_dynamic_sampling(cfg: 'RLConfig', reward_config: Optional['RewardConfig'] = None) -> None:
    if not cfg.dynamic_sample:
        return
    if reward_config is None or not reward_config.orm:
        raise ValueError(
            'dynamic_sample requires the orm channel (a rule or a reward model) because it filters groups '
            'by reward variance.')
    if cfg.num_generations < 2:
        raise ValueError('dynamic_sample requires num_generations >= 2 to measure reward variance.')
    if cfg.max_resample_times < 1:
        raise ValueError('max_resample_times must be >= 1 when dynamic_sample is enabled.')


def _check_reference_sync(cfg: 'RLConfig', train_config: 'TrainConfig',
                          distributed_config: Optional['DistributedConfig'] = None) -> None:
    if not cfg.sync_ref_model:
        return
    if cfg.beta in (None, 0, 0.0):
        raise ValueError('sync_ref_model requires beta > 0 and an active reference model.')
    if cfg.ref_model_sync_steps < 1:
        raise ValueError('ref_model_sync_steps must be >= 1.')
    if not 0.0 < cfg.ref_model_mixup_alpha <= 1.0:
        raise ValueError('ref_model_mixup_alpha must be in (0, 1]; 0 would discard the reference entirely.')
    # Paired EMA requires both sides to have IDENTICAL sharding. When reference_parallel_spec is explicit it
    # must equal the policy's; None auto-inherits and is always valid.
    if cfg.reference_parallel_spec is not None and distributed_config is not None:
        policy_spec = distributed_config.parallel_spec
        if cfg.reference_parallel_spec != policy_spec:
            raise ValueError(
                f'sync_ref_model paired EMA requires reference_parallel_spec to match the policy parallel_spec '
                f'({policy_spec!r}), got {cfg.reference_parallel_spec!r}. Set reference_parallel_spec=None to '
                f'auto-inherit, or use the same spec as the policy.')
    # LoRA's disable_adapter reference shares the policy actor -- EMA against itself is meaningless.
    if cfg.ref_adapters:
        pass  # explicit ref_adapters = a separate frozen copy, EMA is valid
    del train_config


def _check_chord(cfg: 'RLConfig') -> None:
    if not cfg.chord_sft_dataset:
        return
    required = {
        'chord_sft_per_device_train_batch_size': cfg.chord_sft_per_device_train_batch_size,
        'chord_mu_warmup_steps': cfg.chord_mu_warmup_steps,
        'chord_mu_decay_steps': cfg.chord_mu_decay_steps,
        'chord_mu_peak': cfg.chord_mu_peak,
        'chord_mu_valley': cfg.chord_mu_valley,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f'chord_sft_dataset requires explicit CHORD schedule fields: {missing}.')
    if cfg.chord_sft_per_device_train_batch_size < 1:
        raise ValueError('chord_sft_per_device_train_batch_size must be >= 1.')
    if cfg.chord_mu_warmup_steps < 0 or cfg.chord_mu_decay_steps < 0:
        raise ValueError('CHORD warmup and decay steps must be non-negative.')
    if not 0.0 <= cfg.chord_mu_valley <= cfg.chord_mu_peak <= 1.0:
        raise ValueError('CHORD requires 0 <= chord_mu_valley <= chord_mu_peak <= 1.')


def _check_router_replay(cfg: 'RLConfig') -> None:
    """Reject routing-replay combinations dev cannot honour (GRPO-only, never silently dropped).

    The RECORD/REPLAY driving lives in the GRPO assembler; MoE/sampler-export checks surface at runtime.
    """
    mode = cfg.router_replay_mode
    if mode == 'disabled':
        return
    if cfg.rl_type != 'grpo':
        raise ValueError(f'router_replay_mode={mode!r} is GRPO-only: R2/R3 routing replay is driven by '
                         'the GRPO assembler, and the other recipes build their own forward_backward kwargs '
                         'without it. Use rl_type=grpo, or router_replay_mode=disabled.')


def _check_sampling_replay(cfg: 'RLConfig', rollout_config: Optional['RolloutConfig'] = None) -> None:
    """Reject sampling-replay combinations dev cannot honour, so the knob is never silently dropped.

    GRPO-only, no loss-side KL (beta must be 0 unless folded into the reward), no multi-turn.
    """
    if not cfg.enable_sampling_replay:
        return
    if cfg.rl_type != 'grpo':
        raise ValueError(f'enable_sampling_replay is GRPO-only: the sampling_masks column is assembled by '
                         f'the GRPO assembler, but rl_type={cfg.rl_type!r} builds its own forward_backward kwargs '
                         'without it. Use rl_type=grpo, or disable enable_sampling_replay.')
    calculate_kl = cfg.calculate_KL is not False
    loss_beta = cfg.beta if (cfg.beta is not None and not cfg.kl_in_reward and calculate_kl) else None
    if loss_beta:
        raise ValueError(f'enable_sampling_replay forbids a GRPO KL penalty in the loss (beta must be 0), but '
                         f'beta={loss_beta} would reach GRPOLoss. Set beta=0, or fold the KL into the reward '
                         'with kl_in_reward=True, or disable enable_sampling_replay.')
    if rollout_config is not None and rollout_config.max_turns is not None:
        raise ValueError('enable_sampling_replay does not support multi-turn rollouts: tool/observation turns '
                         'are trainable but were not drawn from the sampled policy, so they have no sampling '
                         'mask to replay against (GRPOLoss rejects the resulting loss_mask != trainable). Set '
                         'max_turns=None, or disable enable_sampling_replay.')


def _check_prm(cfg: 'RLConfig', reward_config: Optional['RewardConfig'] = None) -> None:
    """Reject GRPO process-reward (PRM) combinations the loop cannot honour (no silently dead knob).

    Refuses RLSD/SDAR overlap, scorer knobs with no --prm channel, prm_orm_mix outside [0, 1].
    The prm channel and its knobs all live on the RewardConfig; a missing one counts as "no channel".
    """
    if cfg.rl_type != 'grpo':
        return
    scorer_knobs = sorted({'prm_scorer', 'prm_step_delimiter', 'prm_orm_mix'}.intersection(
        _changed_fields(reward_config)))
    if reward_config is None or not reward_config.prm:
        if scorer_knobs:
            details = ', '.join(f'--{name}' for name in scorer_knobs)
            raise ValueError(
                f'{details} shape the GRPO process reward but no --prm channel is set, so they would do '
                'nothing. Add a --prm item (a rule name or a frozen PRM model id), or drop them.')
        return
    if not 0.0 <= reward_config.prm_orm_mix <= 1.0:
        raise ValueError(
            f'prm_orm_mix must be in [0, 1] (the PRM/ORM convex-combination ratio: 0 is standard GRPO, 1 is '
            f'pure process reward), got {reward_config.prm_orm_mix}.')
    if cfg.advantage_reweight == 'rlsd' or cfg.sdar_loss_coef > 0:
        raise ValueError(
            'The --prm process reward cannot be combined with RLSD/SDAR: both turn the scalar advantage into a '
            'per-token one from different sources (process reward vs teacher-signal reweighting), and only one '
            'can drive the loss. Drop --prm, or disable advantage_reweight=rlsd / sdar_loss_coef.')


def _check_self_distillation(cfg: 'RLConfig', train_config: 'TrainConfig',
                            reward_config: Optional['RewardConfig'] = None) -> None:
    _check_rlsd(cfg, reward_config)
    _check_sdar(cfg)
    if (cfg.advantage_reweight == 'rlsd' or cfg.sdar_loss_coef > 0) and train_config.use_liger_kernel:
        raise ValueError('RLSD and SDAR require the unfused per-token loss path; disable use_liger_kernel.')


def _check_rlsd(cfg: 'RLConfig', reward_config: Optional['RewardConfig'] = None) -> None:
    if cfg.advantage_reweight != 'rlsd':
        return
    if not 0.0 <= cfg.rlsd_lambda <= 1.0:
        raise ValueError('rlsd_lambda must be in [0, 1].')
    if cfg.rlsd_reweight_clip_range < 0:
        raise ValueError('rlsd_reweight_clip_range must be >= 0.')
    if cfg.rlsd_lambda_warmup_steps < 0 or cfg.rlsd_lambda_decay_steps < 0:
        raise ValueError('RLSD warmup and decay steps must be non-negative.')
    if reward_config is None or not reward_config.orm:
        raise ValueError('advantage_reweight=rlsd requires the orm channel (a rule or a reward model).')


def _check_sdar(cfg: 'RLConfig') -> None:
    if cfg.sdar_loss_coef <= 0:
        return
    if cfg.sdar_gate_beta <= 0:
        raise ValueError('sdar_gate_beta must be > 0.')
    if cfg.advantage_reweight == 'rlsd':
        raise ValueError('SDAR and RLSD cannot be enabled together.')


def _check_distillation(cfg: 'RLConfig') -> None:
    """Validate the on-policy distillation family (gkd / opsd / mopd).

    lmbda/distinct-teacher shared; JSD knobs GKD-only; OPSD/MOPD refuse non-zero beta (no KL term).
    """
    if cfg.rl_type not in ('gkd', 'opsd', 'mopd'):
        return
    if not 0.0 <= cfg.lmbda <= 1.0:
        raise ValueError(f'{cfg.rl_type} lmbda must be in [0, 1].')
    if cfg.offload_teacher_model and cfg.teacher_model is None:
        raise ValueError('offload_teacher_model requires a distinct local teacher_model.')
    if cfg.teacher_deepspeed and cfg.teacher_model is None:
        raise ValueError('teacher_deepspeed requires a distinct local teacher_model.')
    if cfg.rl_type in ('opsd', 'mopd'):
        if cfg.beta not in (None, 0, 0.0):
            raise ValueError(f'{cfg.rl_type} v1 has no reference-KL term; beta={cfg.beta!r} would be ignored. '
                             'Leave beta unset/0 -- self-distillation pulls the student toward the teacher directly.')
        return
    if cfg.rl_type != 'gkd':
        return
    if cfg.sft_alpha < 0:
        raise ValueError('GKD sft_alpha must be >= 0.')
    if cfg.distill_temperature <= 0:
        raise ValueError('GKD distill_temperature must be > 0.')
    if cfg.gkd_logits_topk is not None and cfg.gkd_logits_topk < 1:
        raise ValueError('GKD gkd_logits_topk must be >= 1 when set.')


def _check_auxiliary_adapters(cfg: 'RLConfig', reward_config: Optional['RewardConfig'] = None) -> None:
    if len(cfg.ref_adapters) > 1:
        raise ValueError('ref_adapters currently supports one frozen reference adapter.')
    if len(cfg.teacher_adapters) > 1:
        raise ValueError('teacher_adapters currently supports one frozen teacher adapter.')
    if cfg.teacher_adapters and cfg.teacher_model is None:
        raise ValueError('teacher_adapters requires a distinct local teacher_model.')
    if reward_config is not None:
        _check_channel_companions(reward_config, 'orm')
        _check_channel_companions(reward_config, 'prm')


def _check_channel_companions(reward_config: 'RewardConfig', channel: str) -> None:
    """Each channel's companion family aligns positionally with its MODEL subsequence.

    One entry per model id (rule items hold no model); the synthesis path's lone-adapter sampler-reuse
    spelling is resolved there (resolve_reward_channel), which validate never sees.
    """
    from swift.dev.rewards.reward import is_registered_reward

    items = getattr(reward_config, channel) or []
    model_count = sum(1 for item in items if not is_registered_reward(item))
    companions = {
        f'{channel}_adapters': getattr(reward_config, f'{channel}_adapters'),
        f'{channel}_model_type': getattr(reward_config, f'{channel}_model_type'),
        f'{channel}_model_revision': getattr(reward_config, f'{channel}_model_revision'),
        f'{channel}_template': getattr(reward_config, f'{channel}_template'),
    }
    for field, values in companions.items():
        if not values:
            continue
        if model_count == 0:
            raise ValueError(f'{field} requires a model id in the {channel} channel.')
        if len(values) != model_count:
            raise ValueError(
                f'{field} must contain exactly one entry per model id in the {channel} channel '
                f'({model_count} model id(s)), got {len(values)}.')


def _check_logging(logging_config: Optional['LoggingConfig']) -> None:
    if logging_config is None:
        return
    reporters = {name.lower() for name in logging_config.report_to}
    supported = {'none', 'tensorboard', 'wandb', 'swanlab'}
    unknown = reporters - supported
    if unknown:
        raise ValueError(f'Unsupported LoggingConfig.report_to values: {sorted(unknown)}.')
    if 'none' in reporters and len(reporters) > 1:
        raise ValueError('LoggingConfig.report_to cannot combine "none" with an active tracker.')
    if logging_config.logging_strategy == 'steps' and logging_config.logging_steps <= 0:
        raise ValueError('LoggingConfig.logging_steps must be > 0 when logging_strategy="steps".')
    if logging_config.swanlab_notification_method == 'email':
        required = (
            logging_config.swanlab_sender_email,
            logging_config.swanlab_receiver_email,
            logging_config.swanlab_smtp_server,
            logging_config.swanlab_smtp_port,
        )
        if not all(required):
            raise ValueError('SwanLab email notification requires sender_email, receiver_email, smtp_server, and '
                             'smtp_port.')


_GALORE_UNSUPPORTED = (
    'galore_optim_per_parameter',
    'galore_with_embedding',
)


def _check_galore(train_config: 'TrainConfig', tuner_config: Optional['TunerConfig']) -> None:
    """Refuse GaLore combinations that cannot train the way the user asked.

    Fatal: with an adapter, unsupported legacy knobs, or a base optim with no GaLore variant.
    """
    if not train_config.use_galore:
        return
    if tuner_config is not None:
        raise ValueError(
            f'use_galore cannot be combined with tuner={tuner_config.tuner!r}: GaLore projects the '
            'full-parameter gradient into a low-rank subspace, a different low-rank approximation than '
            'the adapter. Train full-param with GaLore (--tuner full --use_galore true), or use the '
            'adapter without GaLore.')
    unsupported = [name for name in _GALORE_UNSUPPORTED if name in _changed_fields(train_config)]
    if unsupported:
        raise NotImplementedError(
            f'{unsupported} are legacy GaLore extensions that twinkle\'s GaLoreConfig does not implement '
            '(it projects the full-precision gradient with the rank / target_modules / update_proj_gap / '
            'scale / proj_type knobs, plus QGaLore quantized projection via galore_quantization). Remove '
            'them, or keep use_galore to the supported fields.')
    from swift.dev.utils.naming import resolve_galore_target, resolve_optim_target
    try:
        resolve_galore_target(resolve_optim_target(train_config.optim)[0], quantize=train_config.galore_quantization)
    except NotImplementedError as exc:
        raise ValueError(str(exc)) from exc


def _target_is_all_linear(tuner_config: 'TunerConfig') -> bool:
    """Whether the adapter target resolves to 'all-linear' (the worker-side MM expansion trigger).

    An explicit target_regex/module list bypasses the expansion, so freeze_* flags are inert there.
    """
    if tuner_config.target_regex:
        return tuner_config.target_regex == 'all-linear'
    return 'all-linear' in (tuner_config.target_modules or [])


def _check_param_groups(tuner_config: Optional['TunerConfig']) -> None:
    """Fail loudly on param_groups knobs that cannot apply the way the user asked.

    LoRA+ needs a LoRA-family adapter; freeze_llm/aligner steer only the 'all-linear' MM expansion.
    """
    if tuner_config is None:
        return
    if tuner_config.lorap_lr_ratio is not None and tuner_config.tuner != 'lora':
        raise ValueError(
            f'lorap_lr_ratio (LoRA+) scales the lora_B / embedding learning rate against lora_A, so it '
            f'needs a LoRA-family adapter, but tuner={tuner_config.tuner!r} has no lora_B parameters. '
            'Use tuner="lora", or drop lorap_lr_ratio / lorap_emb_lr.')
    if set(_changed_fields(tuner_config)) & {'freeze_llm', 'freeze_aligner'} and not _target_is_all_linear(tuner_config):
        logger.warning(
            'freeze_llm / freeze_aligner were set but the adapter target is an explicit '
            'target_regex / module list, not "all-linear". These flags only steer the multimodal '
            '"all-linear" target expansion, so they are ignored here and the run trains exactly the '
            'modules the explicit target names.')


def _check_muon(train_config: 'TrainConfig', distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Reject muon pairings that Megatron itself refuses, or that would train something else.

    Break from legacy: raises on the distributed optimizer instead of silently turning it off.
    """
    if not is_megatron:
        # Point a Megatron `optimizer='muon'` mistake at the transformers spelling instead of letting
        # it be silently ignored (the Megatron branch is the only one that reads `optimizer`).
        if 'muon' in train_config.optimizer:
            raise ValueError(
                f'TrainConfig.optimizer={train_config.optimizer!r} is a Megatron optimizer, but the active '
                'backend is transformers. Use TrainConfig.optim for the transformers path, or switch '
                'DistributedConfig.backend.')
        # muon_momentum/use_nesterov/num_ns_steps are dual-backend, qk_clip_* transformers-only; all are
        # read on this backend ONLY when optim='muon', so guard them against being silently ignored.
        if train_config.optim.lower() != 'muon':
            stray = [n for n in ('muon_momentum', 'muon_use_nesterov', 'muon_num_ns_steps',
                                 'qk_clip_enabled', 'qk_clip_tau')
                     if n in _changed_fields(train_config)]
            if stray:
                raise ValueError(
                    f'{stray} configure the muon optimizer, but TrainConfig.optim={train_config.optim!r} '
                    "on the transformers backend, so they would be ignored. Set optim='muon' to use "
                    'them, or drop them.')
        return

    if 'muon' not in train_config.optimizer:
        return

    from swift.dev.utils.naming import mcore_version_at_least
    if not mcore_version_at_least('0.16'):
        raise ValueError(f'TrainConfig.optimizer={train_config.optimizer!r} requires megatron-core>=0.16, which is '
                         'where the muon implementation lands.')

    if train_config.optimizer == 'muon':
        # Plain muon orthogonalises whole parameters, so it needs each one gathered before the step;
        # both overlaps hand it a shard instead. megatron asserts the same pairing.
        for attr in ('overlap_grad_reduce', 'overlap_param_gather'):
            if getattr(distributed_config, attr):
                raise ValueError(
                    f"optimizer='muon' is incompatible with DistributedConfig.{attr}=True: muon computes its "
                    'update from the whole parameter, which an overlapped reduce/gather has not finished '
                    f"assembling. Use optimizer='dist_muon', which is sharded, or set {attr}=False.")

    if distributed_config.use_distributed_optimizer:
        raise ValueError(f'TrainConfig.optimizer={train_config.optimizer!r} does not support '
                         'DistributedConfig.use_distributed_optimizer=True; muon maintains its own state layout. '
                         'legacy turned the distributed optimizer off silently here -- set it to False explicitly, '
                         'so the memory profile and checkpoint contents of the run are not a surprise.')


def _check_unsloth_strategy(distributed_config: 'DistributedConfig',
                            template_config: 'TemplateConfig') -> None:
    """Refuse the strategies unsloth (backend='unsloth') cannot co-exist with.

    It rebuilds the graph for whole, unsharded weights: ZeRO/FSDP and Ulysses SP are refused.
    """
    if distributed_config.backend != 'unsloth':
        return
    if distributed_config.deepspeed:
        raise NotImplementedError(
            'backend="unsloth" cannot run under DeepSpeed: unsloth installs its own kernels and module '
            'graph and does not compose with a ZeRO strategy. Drop --deepspeed, or use the default backend.')
    if distributed_config.fsdp:
        raise NotImplementedError(
            'backend="unsloth" cannot run under FSDP: unsloth installs its own kernels and module graph '
            'and does not compose with parameter sharding. Drop --fsdp, or use the default backend.')
    if template_config.sequence_parallel_size > 1:
        raise NotImplementedError(
            f'backend="unsloth" cannot run under sequence_parallel_size='
            f'{template_config.sequence_parallel_size}: unsloth\'s fused kernels assume a whole sequence per '
            'rank. Set sequence_parallel_size=1, or use the default backend.')


def _check_deepspeed_autotp(distributed_config: 'DistributedConfig') -> None:
    """DeepSpeed AutoTP is not implemented on the twinkle strategy, so refuse the knob rather than drop it.

    AutoTP needs TP groups AND a TP-aware sampler; half-wiring it feeds TP peers different data.
    """
    if distributed_config.deepspeed_autotp_size is None:
        return
    raise NotImplementedError(
        f'deepspeed_autotp_size={distributed_config.deepspeed_autotp_size} (DeepSpeed AutoTP) is not implemented '
        'by the twinkle DeepSpeed strategy: AutoTP needs tensor-parallel groups plus a TP-aware data sampler, and '
        'only the ZeRO config side exists here. Drop it and rely on ZeRO sharding (the deepspeed presets), or use '
        'the megatron backend for tensor parallelism.')


def _check_eval_generation(train_config: 'TrainConfig', template_config: 'TemplateConfig',
                           distributed_config: 'DistributedConfig') -> None:
    """Guards for the in-training generative eval path (predict_with_generate -> EvalScope).

    Refuses no eval_dataset, SP transformers sampler, vllm/sglang outside ray, sharded multi-rank local.
    """
    if not train_config.predict_with_generate:
        return
    if not train_config.eval_dataset:
        raise ValueError(
            'predict_with_generate=True runs an EvalScope benchmark inside training, so it needs '
            'TrainConfig.eval_dataset; got none. Pass --eval_dataset <name>, or use the validation-loss path '
            '(--predict_with_generate false) which evaluates the split-off validation set.')
    backend = train_config.eval_sampler_backend
    if backend == 'transformers' and template_config.sequence_parallel_size > 1:
        raise ValueError(
            f'predict_with_generate=True with eval_sampler_backend="transformers" cannot run under '
            f'sequence_parallel_size={template_config.sequence_parallel_size}: HF .generate() runs the whole '
            'sequence on one rank and is unaware of the ulysses/ring shard, so it would generate from a partial '
            'sequence. Evaluate with the validation-loss path (--predict_with_generate false), or serve generation '
            'from a co-resident engine (--eval_sampler_backend vllm|sglang, which needs --mode ray).')
    if backend in ('vllm', 'sglang') and distributed_config.mode != 'ray':
        raise NotImplementedError(
            f'eval_sampler_backend={backend!r} weight-syncs the training weights into a co-resident engine through '
            f'a Ray DeviceGroup (CheckpointEngineManager), which mode={distributed_config.mode!r} cannot place. Run '
            'under --mode ray, or use --eval_sampler_backend transformers (which wraps the live training module).')
    # rank0-only generation is safe only while the model's forward is rank-local. Only ZeRO-3 shards params
    # (ZeRO-1/2 and DDP replicate them and stay rank-local), so those are allowed.
    if backend == 'transformers' and distributed_config.mode == 'local':
        from swift.dev.utils import deepspeed_zero_stage, get_dist_setting
        world_size = get_dist_setting()[2]
        shards_params = bool(distributed_config.fsdp) or deepspeed_zero_stage(distributed_config.deepspeed) == 3
        if shards_params and world_size > 1:
            sharding = 'FSDP (--fsdp)' if distributed_config.fsdp else 'DeepSpeed ZeRO-3 (--deepspeed)'
            raise ValueError(
                f'predict_with_generate=True with eval_sampler_backend="transformers" drives generation from rank 0 '
                f'only, but the transformers path shards parameters under {sharding} across {world_size} ranks: '
                "rank 0's .generate() all-gathers weights that its peers -- parked at the eval barrier -- never "
                'join, so the run hangs at the first generative eval. Run it under --mode ray (the driver '
                'dispatches generate across the group), or serve generation from a co-resident engine '
                '(--eval_sampler_backend vllm|sglang, which needs --mode ray), or use a replicated strategy '
                '(plain DDP, or DeepSpeed ZeRO-1/2), or the validation-loss path (--predict_with_generate false).')


def _check_megatron_fsdp(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Reject Megatron-FSDP pairings that megatron itself rejects, or that silently do nothing.

    Duplicated with MegatronStrategy._check_fsdp (the authority) so a CLI typo fails on the driver.
    """
    if not distributed_config.use_megatron_fsdp:
        return

    if not is_megatron:
        # The transformers backend has its own FSDP, reached through DistributedConfig.fsdp. Silently
        # ignoring this flag there would leave a run that says "sharded" and replicates.
        raise ValueError('DistributedConfig.use_megatron_fsdp only applies to the megatron backend, but the active '
                         'backend is transformers. Use DistributedConfig.fsdp for the transformers path.')

    if not distributed_config.use_distributed_optimizer:
        raise ValueError('DistributedConfig.use_megatron_fsdp requires use_distributed_optimizer=True: FSDP shards '
                         'the parameters, and only the distributed optimizer keeps the matching master-weight '
                         'shards to update them from.')

    if distributed_config.context_parallel_size > 1:
        # megatron asserts the same pairing on its own CLI ('Hybrid context parallelism not supported
        # with Megatron FSDP').
        raise ValueError('DistributedConfig.use_megatron_fsdp is incompatible with context_parallel_size='
                         f'{distributed_config.context_parallel_size}. Megatron-FSDP does not support context '
                         'parallelism; use the default DDP wrapper for a CP run.')


# (format field, param-gather field) per low-precision format. The amax knobs are absent on purpose:
# non-None defaults make "did the user set this?" unanswerable, so a dependency check would fire always.
_QUANT_FORMATS = (('fp4_format', 'fp4_param_gather'), ('fp8_format', 'fp8_param_gather'))


def _check_quantization(model_config: 'ModelConfig', distributed_config: 'DistributedConfig',
                        is_megatron: bool) -> None:
    """Reject FP4/FP8 settings that cannot do what they say (errors, not warnings).

    Hardware/TE preconditions are NOT checked here (the driver is not the model-building process).
    """
    active = [fmt for fmt, _ in _QUANT_FORMATS if getattr(model_config, fmt) is not None]
    explicit = getattr(model_config, '_explicit_fields', set())
    format_dependents = {
        'fp4_format': ('fp4_recipe', 'fp4_param_gather'),
        'fp8_format': ('fp8_recipe', 'fp8_amax_history_len', 'fp8_amax_compute_algo', 'fp8_param_gather'),
    }

    for fmt, param_gather in _QUANT_FORMATS:
        if getattr(model_config, fmt) is None:
            requested = [name for name in format_dependents[fmt] if name in explicit]
            if requested:
                raise ValueError(f'ModelConfig fields {requested} need ModelConfig.{fmt} to be set. Without it the '
                                 'model is built in its normal dtype, so these knobs would be ignored.')
            if getattr(model_config, param_gather):
                raise ValueError(f'ModelConfig.{param_gather} needs ModelConfig.{fmt} to be set. Without it the '
                                 'model is built in its normal dtype, so this knob would be ignored.')
            continue

        if not is_megatron:
            raise ValueError(f'ModelConfig.{fmt} is only implemented by the megatron backend, but the active '
                             'backend is transformers. Low-precision training here is a Megatron/Transformer-'
                             'Engine feature; the HF path has no equivalent.')

        if getattr(model_config, param_gather) and not distributed_config.use_distributed_optimizer:
            # Only the distributed optimizer re-quantizes the FP32 master shards back into the quantized
            # parameters; under any other optimizer they never change -- it neither errors nor learns.
            raise ValueError(
                f'ModelConfig.{param_gather} requires DistributedConfig.use_distributed_optimizer=True: '
                'quantized parameters are updated by re-quantizing the distributed optimizer\'s master shards, '
                'and no other optimizer implements that step, so the model would never change.')

    if len(active) > 1:
        # megatron enters exactly one quantization context per transformer layer (its own config raises
        # too); caught here so it fails on the driver, before a model is built on every rank.
        raise ValueError(f'{" and ".join(f"ModelConfig.{fmt}" for fmt in active)} are mutually exclusive: megatron '
                         'applies a single quantization recipe per transformer layer. Pick one.')


def _check_load_quantization(model_config: 'ModelConfig', distributed_config: 'DistributedConfig',
                             tuner_config: Optional['TunerConfig'], quantize_config: Optional['QuantizeConfig'],
                             is_megatron: bool, *, training: bool) -> None:
    """Validate training-time model loading quantization before a worker loads weights."""
    if quantize_config is None or quantize_config.quant_method is None:
        return

    from swift.dev.builders.quantization import CALIBRATION_QUANT_METHODS, LOAD_TIME_QUANT_METHODS

    method = quantize_config.quant_method
    if method in CALIBRATION_QUANT_METHODS:
        if training:
            raise ValueError(
                f'quant_method={method!r} calibrates and exports weights; it is not a training load-time method. '
                'Train from an already quantized checkpoint, or use bnb/hqq/eetq/quanto/fp8 for loading.')
        return
    if method not in LOAD_TIME_QUANT_METHODS:
        raise ValueError(f'Unknown training load-time quant_method={method!r}.')
    if is_megatron:
        raise ValueError(
            f'quant_method={method!r} is a transformers load-time quantizer and is not supported by the Megatron '
            'backend. Use ModelConfig.fp4_format/fp8_format for Transformer-Engine training quantization, or '
            'convert a pre-quantized checkpoint to mcore first.')

    tuner = getattr(tuner_config, 'tuner', 'full') if tuner_config is not None else 'full'
    if training and tuner == 'full':
        raise ValueError(
            f'quant_method={method!r} cannot be combined with full-parameter training: load-time quantized base '
            'weights are not trainable parameters. Select a trainable adapter such as --tuner lora.')

    bits = quantize_config.quant_bits
    valid_bits = {
        'bnb': {4, 8},
        'hqq': {1, 2, 3, 4, 8},
        'eetq': {8},
        'quanto': {2, 4, 8, 'float8'},
        'fp8': {None, 8, 'float8'},
    }[method]
    if bits not in valid_bits:
        raise ValueError(f'quant_method={method!r} does not support quant_bits={bits!r}; expected {sorted(valid_bits, key=str)}.')

    if distributed_config.backend == 'unsloth' and method != 'bnb':
        raise ValueError(
            f'Unsloth only exposes load_in_4bit/load_in_8bit for BNB, so quant_method={method!r} is unsupported. '
            'Use --quant_method bnb or the default backend.')

    if distributed_config.fsdp and method == 'bnb' and bits == 4:
        storage = quantize_config.bnb_4bit_quant_storage
        if storage is None:
            raise ValueError(
                'FSDP QLoRA requires --bnb_4bit_quant_storage to match the model parameter dtype '
                f'(--torch_dtype {model_config.torch_dtype!r}); the bitsandbytes uint8 default cannot be sharded.')
        if model_config.torch_dtype is not None and storage != model_config.torch_dtype:
            raise ValueError(
                f'FSDP QLoRA requires bnb_4bit_quant_storage ({storage!r}) to match torch_dtype '
                f'({model_config.torch_dtype!r}) so flattened FSDP parameters have one dtype.')


def _check_mtp(model_config: 'ModelConfig', is_megatron: bool, tuner_config: Optional['TunerConfig']) -> None:
    """Reject MTP settings that cannot do what they say.

    Failures would otherwise be silent; only LoRA (which can cover the MTP modules) warns instead.
    """
    mtp_dependents = (
        'mtp_loss_scaling_factor', 'enable_mtp_training', 'mtp_freeze', 'mtp_decoder_input_detach',
        'mtp_shared_weights')

    if model_config.mtp_num_layers is None:
        for attr in mtp_dependents:
            value = getattr(model_config, attr)
            if value:
                raise ValueError(f'ModelConfig.{attr}={value!r} needs ModelConfig.mtp_num_layers to be set. '
                                 'Without it no MTP block is built, so this knob would be ignored.')
        return

    if not is_megatron:
        raise ValueError('ModelConfig.mtp_num_layers is only implemented by the megatron backend, but the active '
                         'backend is transformers. MTP lives in mcore-bridge; the HF path has no equivalent.')

    if model_config.mtp_num_layers < 1:
        raise ValueError(f'ModelConfig.mtp_num_layers={model_config.mtp_num_layers} must be >= 1, or None to '
                         'disable MTP entirely.')

    if model_config.mtp_freeze and model_config.enable_mtp_training:
        raise ValueError('ModelConfig.mtp_freeze and ModelConfig.enable_mtp_training are contradictory: the first '
                         'drops the MTP gradient, the second asks for it. Set exactly one.')

    if model_config.mtp_loss_scaling_factor is not None and not model_config.enable_mtp_training:
        raise ValueError('ModelConfig.mtp_loss_scaling_factor only has an effect with '
                         'ModelConfig.enable_mtp_training=True; on its own the MTP loss is never computed, so the '
                         'factor scales nothing.')

    if model_config.mtp_decoder_input_detach and not model_config.enable_mtp_training:
        raise ValueError('ModelConfig.mtp_decoder_input_detach describes where the MTP gradient stops, so it needs '
                         'ModelConfig.enable_mtp_training=True to mean anything.')

    if (model_config.enable_mtp_training and tuner_config is not None
            and getattr(tuner_config, 'tuner', 'full') != 'full'):
        # Not an error: trainable if the adapter targets the MTP modules, which target_modules alone
        # cannot decide. twinkle re-checks against the built model and warns if nothing ended up trainable.
        logger.warning('enable_mtp_training with tuner=%r: the MTP layers are base parameters, so they only '
                       'train if the adapter covers them. Otherwise the MTP loss is computed and discarded.',
                       tuner_config.tuner)


def _check_megatron_attn_backend(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                                 is_megatron: bool) -> None:
    """padding_free needs a variable-length (THD) attention kernel (megatron path).

    legacy silently downgrades here; dev raises deliberately (no softmax_type field, guard unreachable).
    """
    if not is_megatron or not template_config.padding_free:
        return
    if model_config.attn_impl is None:
        return
    # Resolve first: 'sdpa' also lands on the unfused kernel, so comparing the raw string would miss it;
    # unknown values are resolve_megatron_attn_backend's business (better message, at build time).
    from megatron.core.transformer.enums import AttnBackend

    from swift.dev.utils.naming import resolve_megatron_attn_backend
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


# attn_impl values whose kernel handles the variable-length (THD) layout padding_free produces, on the
# transformers backend -- under Ulysses SP and (mirroring legacy sft_args) also without it.
_SP_PADDING_FREE_ATTN_IMPLS = ('flash_attn', 'flash_attention_2', 'flash_attention_3', 'flash_attention_4')


def _check_padding_free_attn(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                             dataset_config: 'DatasetConfig', is_megatron: bool) -> None:
    """padding_free/packing requires a flash attention kernel on the transformers backend.

    Mirrors legacy sft_args._check_padding_free (dev dropped it); None is unsupported too.
    """
    if is_megatron or not (template_config.padding_free or dataset_config.packing):
        return
    feature = 'packing' if dataset_config.packing else 'padding_free'
    if model_config.attn_impl not in _SP_PADDING_FREE_ATTN_IMPLS:
        raise ValueError(
            f'The "{feature}" feature requires a flash attention implementation, got '
            f'attn_impl={model_config.attn_impl!r}. Please use one of: '
            f'{", ".join(_SP_PADDING_FREE_ATTN_IMPLS)}.')


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
    # Streaming has no random access and no precomputed `lengths` column, so the sampler cannot group --
    # previously this died later as an opaque 'lengths must be provided'.
    if dataset_config.streaming:
        raise ValueError('group_by_length requires a map-style dataset: streaming datasets have no `lengths` '
                         'column and no random access, so samples cannot be reordered by length. '
                         'Set streaming=False or group_by_length=False.')
    # `lengths` only exists after an EAGER encode; lazy_tokenize=None is auto, and auto always backs off
    # to eager when group_by_length is on, so only an EXPLICIT opt-in to lazy is a conflict here.
    if dataset_config.lazy_tokenize:
        raise ValueError('group_by_length requires lazy_tokenize=False: the per-sample `lengths` column it '
                         'sorts on is only produced by eager encoding (AddLengthPreprocessor).')


def _check_lazy_tokenize(dataset_config: 'DatasetConfig') -> None:
    """Explicit lazy_tokenize=True conflicts with packing / streaming (legacy base_args.py:136-140).

    None is auto, and auto backs off to eager when either is on, so only the explicit opt-in conflicts.
    """
    if not dataset_config.lazy_tokenize:
        return
    if dataset_config.packing:
        raise ValueError('packing and lazy_tokenize are incompatible: PackingDataset reads the '
                         '`lengths` column at construction (packing.py:78), which only eager '
                         'encoding writes.')
    if dataset_config.streaming:
        raise ValueError('streaming and lazy_tokenize are incompatible.')


def _check_data_sharding(dataset_config: 'DatasetConfig') -> None:
    """data_sharding needs a shuffled order to reshuffle; it is a no-op under sequential reads.

    The group_by_length conflict is NOT fatal here: build_dataset performs legacy's warning downgrade.
    """
    if dataset_config.data_sharding and not dataset_config.train_dataloader_shuffle:
        raise ValueError('data_sharding requires train_dataloader_shuffle=True: it only changes the SCOPE of the '
                         'per-epoch reshuffle (shuffle within a rank shard vs. globally), so with shuffling off '
                         'it does nothing.')


def _check_streaming(dataset_config: 'DatasetConfig') -> None:
    """Refuse the dataset combinations a stream cannot serve.

    Resume works (skip_consumed_samples); only the epoch count is unrestorable after a resume.
    """
    # A cached_dataset is a map-style Dataset written by save_to_disk, so it cannot participate in the
    # streaming pipeline. Legacy asserts the same in SwiftSft._prepare_dataset.
    if dataset_config.streaming and (dataset_config.cached_dataset or dataset_config.cached_val_dataset):
        raise ValueError('cached_dataset does not support streaming=True: the exported cache is a map-style '
                         'dataset loaded via load_from_disk. Set streaming=False, or drop cached_dataset.')


def _check_megatron_recompute(train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                              is_megatron: bool) -> None:
    """gradient_checkpointing decides nothing on Megatron; recompute_granularity does.

    flag-off-yet-recompute-configured is fatal; flag-on-but-unconfigured is the default (warn only).
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


def _check_eval_iters(train_config: 'TrainConfig') -> None:
    if train_config.eval_iters == -1 or train_config.eval_iters > 0:
        return
    raise ValueError('TrainConfig.eval_iters must be -1 (evaluate the full dataset) or a positive batch count.')


def _check_megatron_microbatch_schedule(train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                                         is_megatron: bool) -> None:
    """Validate the VPP micro-batch group against the loop's actual micro-batch count."""
    group = train_config.microbatch_group_size_per_vp_stage
    if group is None:
        return
    if not is_megatron:
        raise ValueError('microbatch_group_size_per_vp_stage is only implemented by the megatron backend.')

    vpp = distributed_config.virtual_pipeline_model_parallel_size
    if vpp is None:
        raise ValueError('microbatch_group_size_per_vp_stage requires virtual_pipeline_model_parallel_size or '
                         'pipeline_model_parallel_layout; a non-interleaved pipeline does not consume this option.')
    if group <= 0:
        raise ValueError('microbatch_group_size_per_vp_stage must be > 0.')

    pp = distributed_config.pipeline_model_parallel_size
    num_microbatches = train_config.gradient_accumulation_steps
    if group < pp or group > num_microbatches:
        raise ValueError(f'microbatch_group_size_per_vp_stage={group} must be in '
                         f'[pipeline_model_parallel_size={pp}, gradient_accumulation_steps={num_microbatches}].')
    remainder = num_microbatches % group
    if 0 < remainder < pp:
        raise ValueError(f'gradient_accumulation_steps % microbatch_group_size_per_vp_stage is {remainder}, which '
                         f'must be 0 or at least pipeline_model_parallel_size={pp}.')


def _check_megatron_optimizer(train_config: 'TrainConfig', is_megatron: bool) -> None:
    """Reject an inconsistent Megatron optimizer/scheduler config before the weights are loaded."""
    if not is_megatron:
        return
    from swift.dev.builders.optimizer import megatron_weight_decay_bounds, warmup_budget

    # Delegates so the weight-decay rule has one definition (configure_optimizer uses the same).
    start_wd, end_wd = megatron_weight_decay_bounds(train_config)
    if start_wd < 0 or end_wd < start_wd:
        raise ValueError(f'Megatron weight decay requires 0 <= start_weight_decay <= end_weight_decay; got '
                         f'{start_wd} -> {end_wd}.')
    if train_config.learning_rate <= 0:
        raise ValueError('Megatron learning_rate must be > 0.')
    if not 0 <= train_config.min_lr <= train_config.learning_rate:
        raise ValueError('Megatron min_lr must satisfy 0 <= min_lr <= learning_rate.')
    if not 0 <= train_config.lr_warmup_init <= train_config.learning_rate:
        raise ValueError('Megatron lr_warmup_init must satisfy 0 <= lr_warmup_init <= learning_rate.')

    # 'cosine_with_min_lr' is Megatron's plain cosine plus a floor, so without min_lr it would run as
    # ordinary cosine -- the name silently not doing what it says.
    if train_config.lr_scheduler.lower() == 'cosine_with_min_lr' and not train_config.min_lr:
        raise ValueError("lr_scheduler='cosine_with_min_lr' needs TrainConfig.min_lr > 0 on the Megatron "
                         'backend; with min_lr=0 it is just cosine. Set min_lr, or use '
                         "lr_scheduler='cosine'.")

    decay_steps = train_config.lr_decay_iters
    if decay_steps is not None and decay_steps <= 0:
        raise ValueError('Megatron lr_decay_iters must be > 0 when set.')
    if decay_steps is None and train_config.max_steps > 0:
        decay_steps = train_config.max_steps
    if decay_steps is not None:
        warmup_steps = warmup_budget(train_config, decay_steps, is_megatron=True)
        if warmup_steps >= decay_steps:
            raise ValueError(
                f'Megatron warmup ({warmup_steps} steps) must be shorter than lr decay ({decay_steps} steps).')

    style = train_config.lr_decay_style
    explicit = getattr(train_config, '_explicit_fields', set())
    wsd_style_explicit = 'lr_wsd_decay_style' in explicit
    if style == 'WSD':
        if train_config.lr_wsd_decay_iters is None or train_config.lr_wsd_decay_iters <= 0:
            raise ValueError("lr_decay_style='WSD' requires lr_wsd_decay_iters > 0.")
        if decay_steps is not None and train_config.lr_wsd_decay_iters > decay_steps:
            raise ValueError('lr_wsd_decay_iters cannot exceed the effective lr decay horizon.')
    elif train_config.lr_wsd_decay_iters is not None or wsd_style_explicit:
        raise ValueError('lr_wsd_decay_iters/lr_wsd_decay_style only apply when lr_decay_style="WSD".')

    _check_optimizer_specific_fields(train_config)


def _check_optimizer_specific_fields(train_config: 'TrainConfig') -> None:
    """Reject optimizer options that the selected Megatron optimizer would ignore."""
    import dataclasses

    explicit = getattr(train_config, '_explicit_fields', set())
    defaults = {field.name: field.default for field in dataclasses.fields(train_config)}

    def changed(name: str) -> bool:
        return name in explicit or getattr(train_config, name) != defaults[name]

    muon_fields = (
        'muon_momentum', 'muon_split_qkv', 'muon_use_nesterov', 'muon_scale_mode',
        'muon_fp32_matmul_prec', 'muon_coefficient_type', 'muon_num_ns_steps', 'muon_tp_mode',
        'muon_extra_scale_factor', 'muon_scalar_optimizer')
    if 'muon' not in train_config.optimizer:
        invalid = [name for name in muon_fields if changed(name)]
        if invalid:
            raise ValueError(f'Muon optimizer fields require optimizer="muon" or "dist_muon": {invalid}.')
    if train_config.optimizer != 'sgd' and changed('sgd_momentum'):
        raise ValueError('sgd_momentum only applies when optimizer="sgd".')
    if train_config.optimizer != 'adam':
        invalid = [name for name in ('adam_beta1', 'adam_beta2', 'adam_epsilon') if changed(name)]
        if invalid:
            raise ValueError(f'Adam optimizer fields only apply when optimizer="adam": {invalid}.')

    precision_fields = ('main_params_dtype', 'main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype')
    if not train_config.use_precision_aware_optimizer:
        invalid = [name for name in precision_fields if changed(name)]
        if invalid:
            raise ValueError(
                f'Precision-aware optimizer dtypes require use_precision_aware_optimizer=True: {invalid}.')
    if not 0 <= train_config.optimizer_offload_fraction <= 1:
        raise ValueError('optimizer_offload_fraction must be in [0, 1].')
    if not train_config.optimizer_cpu_offload and changed('optimizer_offload_fraction'):
        raise ValueError('optimizer_offload_fraction only applies when optimizer_cpu_offload=True.')


def _is_off(value, off_value) -> bool:
    if off_value is None:
        # A falsy NUMBER is a real setting, a falsy container is not: start_weight_decay=0.0 must count
        # as set, while the CLI's [] for an unset fsdp must still count as unset.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return False
        return not value
    if not off_value:
        return not value
    return value == off_value


# (holder, field, off-value) triples rejected on the transformers backend. clip_grad is NOT here (CLI
# folds it into max_grad_norm); the dual-backend muon knobs are excluded too (_check_muon guards them).
_MEGATRON_ONLY = (
    ('model_config', 'vit_attn_impl', None),
    ('model_config', 'language_model_only', False),
    ('dataset_config', 'data_sharding', False),
    ('train_config', 'weight_decay_incr_style', 'constant'),
    ('train_config', 'start_weight_decay', None),
    ('train_config', 'end_weight_decay', None),
    ('train_config', 'min_lr', 0.0),
    ('train_config', 'optimizer', 'adam'),
    ('train_config', 'sgd_momentum', 0.9),
    ('train_config', 'muon_split_qkv', True),
    ('train_config', 'muon_scale_mode', 'spectral'),
    ('train_config', 'muon_fp32_matmul_prec', 'medium'),
    ('train_config', 'muon_coefficient_type', 'quintic'),
    ('train_config', 'muon_tp_mode', 'blockwise'),
    ('train_config', 'muon_extra_scale_factor', 1.0),
    ('train_config', 'muon_scalar_optimizer', 'adam'),
    ('train_config', 'use_precision_aware_optimizer', False),
    ('train_config', 'main_params_dtype', 'fp32'),
    ('train_config', 'main_grads_dtype', 'fp32'),
    ('train_config', 'exp_avg_dtype', 'fp32'),
    ('train_config', 'exp_avg_sq_dtype', 'fp32'),
    ('train_config', 'optimizer_cpu_offload', False),
    ('train_config', 'optimizer_offload_fraction', 1.0),
    ('train_config', 'optimizer_cuda_graph', False),
    ('train_config', 'accumulate_allreduce_grads_in_fp32', False),
    ('train_config', 'apply_wd_to_qk_layernorm', False),
    ('train_config', 'global_batch_size', None),
    ('train_config', 'microbatch_group_size_per_vp_stage', None),
    ('train_config', 'calculate_per_token_loss', None),
    ('train_config', 'lr_decay_style', 'cosine'),
    ('train_config', 'lr_decay_iters', None),
    ('train_config', 'lr_warmup_init', 0.0),
    ('train_config', 'lr_wsd_decay_iters', None),
    ('train_config', 'lr_wsd_decay_style', 'exponential'),
    ('distributed_config', 'bridge_backend', 'mcore-bridge'),
    ('distributed_config', 'tensor_model_parallel_size', 1),
    ('distributed_config', 'pipeline_model_parallel_size', 1),
    ('distributed_config', 'context_parallel_size', 1),
    ('distributed_config', 'expert_model_parallel_size', 1),
    ('distributed_config', 'expert_tensor_parallel_size', 1),
    ('distributed_config', 'sequence_parallel', False),
    ('distributed_config', 'use_distributed_optimizer', True),
    ('distributed_config', 'use_megatron_fsdp', False),
    ('distributed_config', 'recompute_granularity', None),
    ('distributed_config', 'recompute_method', None),
    ('distributed_config', 'recompute_num_layers', None),
    ('distributed_config', 'recompute_modules', ['core_attn']),
    ('distributed_config', 'cp_comm_type', None),
    ('distributed_config', 'cp_partition_mode', 'zigzag'),
    ('distributed_config', 'data_parallel_sharding_strategy', 'optim_grads_params'),
    ('distributed_config', 'virtual_pipeline_model_parallel_size', None),
    ('distributed_config', 'pipeline_model_parallel_layout', None),
    ('distributed_config', 'decoder_first_pipeline_num_layers', None),
    ('distributed_config', 'decoder_last_pipeline_num_layers', None),
    ('distributed_config', 'account_for_embedding_in_pipeline_split', False),
    ('distributed_config', 'account_for_loss_in_pipeline_split', False),
    ('distributed_config', 'overlap_grad_reduce', False),
    ('distributed_config', 'overlap_param_gather', False),
    ('distributed_config', 'overlap_param_gather_with_optimizer_step', False),
    ('distributed_config', 'overlap_p2p_comm', True),
    ('distributed_config', 'batch_p2p_comm', None),
    ('distributed_config', 'align_grad_reduce', True),
    ('distributed_config', 'align_param_gather', True),
    ('distributed_config', 'tp_comm_overlap', False),
    ('distributed_config', 'nccl_comm_warmup', False),
)

# Rejected on the megatron backend. qk_clip_* feed twinkle MuonClip's MuonConfig on the transformers
# path only; off_value = the TrainConfig default, so an untouched field is not rejected.
_HF_ONLY = (
    ('model_config', 'experts_impl', None),
    ('model_config', 'max_memory', None),
    ('model_config', 'local_repo_path', None),
    ('model_config', 'model_kwargs', None),
    ('model_config', 'init_strategy', None),
    ('distributed_config', 'deepspeed', None),
    ('distributed_config', 'zero_hpz_partition_size', None),
    ('distributed_config', 'fsdp', None),
    ('distributed_config', 'ddp_find_unused_parameters', None),
    ('distributed_config', 'ddp_broadcast_buffers', None),
    ('distributed_config', 'ddp_bucket_cap_mb', None),
    ('distributed_config', 'ddp_static_graph', None),
    ('train_config', 'use_galore', False),
    ('train_config', 'use_liger_kernel', False),
    ('train_config', 'neftune_noise_alpha', None),
    ('train_config', 'optim', 'adamw_torch_fused'),
    ('train_config', 'optim_args', None),
    ('train_config', 'gradient_checkpointing_kwargs', None),
    ('train_config', 'router_aux_loss_coef', 0.0),
    ('train_config', 'label_smoothing_factor', 0.0),
    ('train_config', 'acc_strategy', 'token'),
    ('train_config', 'predict_with_generate', False),
    ('train_config', 'qk_clip_enabled', False),
    ('train_config', 'qk_clip_tau', 10000.0),
)

_MEGATRON_PARALLEL_SIZES = (
    'tensor_model_parallel_size',
    'pipeline_model_parallel_size',
    'context_parallel_size',
    'expert_model_parallel_size',
)


def _check_backend_specific(model_config: 'ModelConfig',
                            dataset_config: 'DatasetConfig',
                            train_config: 'TrainConfig',
                            distributed_config: 'DistributedConfig',
                            is_megatron: bool,
                            tuner_config: Optional['TunerConfig'] = None) -> None:
    """Reject knobs the active backend does not implement, so they cannot be silently ignored."""
    holders = {
        'model_config': model_config,
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
        explicit = attr in getattr(holder, '_explicit_fields', set())
        if not explicit and _is_off(value, off_value):
            continue
        hint = (f'the {wrong_backend} backend runs with all Megatron parallel sizes == 1'
                if attr in _MEGATRON_PARALLEL_SIZES else f'the active backend is {wrong_backend}')
        raise ValueError(f'{attr}={value!r} is only implemented by the {right_backend} backend, but {hint}. '
                         f'Remove it, or switch DistributedConfig.backend.')


def _check_selective_recompute(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """'selective' recompute chooses WHAT to recompute; recompute_method chooses HOW MUCH of 'full'.

    Selective targets the attention ops with no layer partitioning, so a method beside it is refused.
    """
    if not is_megatron:
        return
    if distributed_config.recompute_granularity == 'selective' and distributed_config.recompute_method is not None:
        raise ValueError('DistributedConfig.recompute_method='
                         f'{distributed_config.recompute_method!r} has no effect with '
                         "recompute_granularity='selective': selective recompute always targets the attention "
                         "ops and has nothing to partition. Use recompute_granularity='full' to configure a "
                         'method, or drop recompute_method.')


def _check_pipeline_decoder_layers(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """The per-stage decoder layer overrides need a pipeline to distribute across (legacy :832-835).

    With pipeline_model_parallel_size == 1 both stages are the whole model, so there is nothing to move.
    """
    if not is_megatron or distributed_config.pipeline_model_parallel_size > 1:
        return
    for attr in ('decoder_first_pipeline_num_layers', 'decoder_last_pipeline_num_layers'):
        if getattr(distributed_config, attr) is not None:
            raise ValueError(f'DistributedConfig.{attr} needs pipeline_model_parallel_size > 1: with a single '
                             'pipeline stage there is no first/last stage to move layers onto. Set a pipeline '
                             f'size, or drop {attr}.')


def _check_freeze_ratio_pp(train_config: 'TrainConfig', distributed_config: 'DistributedConfig', is_megatron: bool,
                           tuner_config: Optional['TunerConfig']) -> None:
    """freeze_parameters_ratio cannot be combined with Megatron pipeline parallelism.

    Under PP>1 each rank holds different layers, so "the first N%" is a meaningless per-stage slice.
    """
    if not is_megatron or tuner_config is not None:
        return
    if train_config.freeze_parameters_ratio and distributed_config.pipeline_model_parallel_size > 1:
        raise ValueError(
            'TrainConfig.freeze_parameters_ratio cannot be combined with Megatron '
            f'pipeline_model_parallel_size={distributed_config.pipeline_model_parallel_size} > 1: each pipeline '
            'rank holds different layers, so freezing a fraction by element count selects a different slice per '
            'stage. Use freeze_parameters / freeze_parameters_regex (name-based, consistent across stages), or '
            'drop pipeline parallelism.')


def _check_tp_comm_overlap(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """TP comm/GEMM overlap only exists when sequence parallelism splits the activations (legacy :896-898).

    The overlap hides the TP collectives behind the GEMM, and they only appear under sequence_parallel.
    """
    if not is_megatron:
        return
    if distributed_config.tp_comm_overlap and not distributed_config.sequence_parallel:
        raise ValueError('DistributedConfig.tp_comm_overlap requires sequence_parallel=True: the overlap hides '
                         'the tensor-parallel collectives that only exist under sequence parallelism, so with it '
                         'off there is nothing to overlap.')


def _check_sequence_parallel_tp(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Sequence parallelism splits activations across the TP ranks, so it needs TP > 1.

    legacy silently turns it off here (:890-891); dev raises -- the memory profile must not be a surprise.
    """
    if not is_megatron:
        return
    if distributed_config.sequence_parallel and distributed_config.tensor_model_parallel_size <= 1:
        raise ValueError('DistributedConfig.sequence_parallel requires tensor_model_parallel_size > 1: it shards '
                         'activations along the sequence across the tensor-parallel ranks, and with TP=1 there are '
                         'none to shard across. legacy turned sequence_parallel off silently here; dev refuses so '
                         'the activation-memory profile of the run is not a surprise. Set a TP size, or '
                         'sequence_parallel=False.')


def _check_checkpoint_runtime(checkpoint_config: Optional['CheckpointConfig'],
                              distributed_config: 'DistributedConfig',
                              tuner_config: Optional['TunerConfig'],
                              rl_config: Optional['RLConfig'],
                              is_megatron: bool, *,
                              training: bool) -> None:
    """Reject checkpoint semantics that the selected loop/backend cannot honor."""
    if not training or checkpoint_config is None:
        return
    changed = set(_changed_fields(checkpoint_config))
    if checkpoint_config.save_strategy != 'steps':
        raise NotImplementedError(
            f'save_strategy={checkpoint_config.save_strategy!r} is not implemented by the Twinkle training loop. '
            'Use --save_strategy steps with --save_steps, or use the legacy CLI.')
    # ``resume_only_model=False`` on the CLI is an explicit full-resume request (the legacy --finetune
    # false spelling folds onto it), so without a resume point the run would silently start from scratch.
    if 'resume_only_model' in changed and not checkpoint_config.resume_only_model \
            and not checkpoint_config.resume_from_checkpoint:
        raise ValueError(
            '--resume_only_model false requests a full resume (optimizer, scheduler, step count), but no '
            '--resume_from_checkpoint is set: the run would silently start from scratch. Point '
            '--resume_from_checkpoint at a checkpoint, or drop the flag.')
    # A per-save hub upload reads the checkpoint files right after save() returns, which is exactly the
    # window async_save leaves the write in flight; the two cannot compose.
    if checkpoint_config.async_save and checkpoint_config.push_to_hub:
        raise ValueError(
            'async_save is incompatible with push_to_hub: the per-save upload reads the checkpoint files '
            'before the background write has landed. Drop one of the two (the final checkpoint can still '
            'be pushed with hub_strategy=end).')
    if not is_megatron:
        unsupported = {'no_save_rng', 'no_load_optim', 'no_load_rng'}.intersection(changed)
        if unsupported:
            raise NotImplementedError(
                f'Transformers checkpoints cannot selectively omit or restore {sorted(unsupported)}. '
                'Use --save_only_model/--resume_only_model, or switch to the Megatron backend.')
        tuner = getattr(tuner_config, 'tuner', 'full') if tuner_config is not None else 'full'
        if tuner != 'full' and 'max_shard_size' in changed:
            raise NotImplementedError(
                'Transformers PEFT adapter checkpoints do not support max_shard_size. Remove the option or use '
                'full-parameter training.')
    else:
        if not checkpoint_config.save_safetensors:
            raise NotImplementedError(
                'The dev Megatron runtime always writes HF-format safetensors checkpoints; '
                'use --save_safetensors true.')
        if not checkpoint_config.safe_serialization:
            raise NotImplementedError('Megatron HF-format checkpoints require --safe_serialization true.')
        if distributed_config.bridge_backend == 'megatron-bridge' and 'max_shard_size' in changed:
            raise NotImplementedError(
                'megatron-bridge AutoBridge does not expose max_shard_size. Use --bridge_backend mcore-bridge or '
                'remove the option.')
    if rl_config is not None and rl_config.rl_type in {'grpo', 'gkd', 'ppo', 'opsd', 'mopd', 'rft'}:
        if 'ignore_data_skip' in changed:
            raise NotImplementedError(
                f'ignore_data_skip does not apply to {rl_config.rl_type} because its online loop has no resumable '
                'dataset iterator. Remove the option.')
        if rl_config.rl_type == 'ppo' and checkpoint_config.save_total_limit == 1:
            raise ValueError(
                'PPO requires save_total_limit >= 2 because the policy and value-model components are saved '
                'sequentially; retaining one previous complete checkpoint avoids a no-valid-checkpoint window.')


def _check_save_total_limit(checkpoint_config: Optional['CheckpointConfig'], is_megatron: bool) -> None:
    """Validate the rolling checkpoint limit and preserve legacy Megatron's stricter lower bound.

    async_save's delete-oldest cannot tell whether a background save finished, so the two conflict.
    """
    if checkpoint_config is None or checkpoint_config.save_total_limit is None:
        return
    if checkpoint_config.save_total_limit < 1:
        raise ValueError('CheckpointConfig.save_total_limit must be >= 1.')
    if checkpoint_config.async_save:
        raise ValueError('CheckpointConfig.save_total_limit is incompatible with async_save=True: the rolling '
                         'delete of old checkpoints cannot tell whether a background save has finished. Disable '
                         'one of the two.')
    if not is_megatron:
        return
    if checkpoint_config.save_total_limit < 2:
        raise ValueError(
            'CheckpointConfig.save_total_limit must be >= 2 on the Megatron backend, matching the legacy CLI.')


# rl_type -> whether it trains against a separate reference model. CPO/ORPO fold the reference into
# their own loss and LoRA uses the adapter-disabled base as reference, so neither takes a ref_model.
_RLHF_USES_REF_MODEL = ('dpo', 'kto', 'ppo', 'grpo')


def _check_rlhf_ref_model(model_config: 'ModelConfig', tuner_config: Optional['TunerConfig'],
                          rl_config: Optional['RLConfig']) -> None:
    """Reject a reference model passed to an algorithm that has none.

    The refusal half of process._derive_rlhf_ref_model: --ref_model on CPO/ORPO/LoRA would be ignored.
    """
    if rl_config is None:
        return
    rl_type = getattr(rl_config, 'rl_type', None)
    tuner = getattr(tuner_config, 'tuner', 'full') if tuner_config is not None else 'full'
    uses_ref = rl_type in _RLHF_USES_REF_MODEL and (tuner == 'full' or bool(rl_config.ref_adapters))
    if rl_config.ref_model is None:
        if rl_config.ref_adapters:
            raise ValueError('ref_adapters requires a reference model; call process_configs or set ref_model.')
        return
    # grpo with beta=0 drops the KL term, so even a ref-using algorithm needs no reference then.
    if rl_type == 'grpo' and rl_config.beta == 0.0:
        uses_ref = False
    if not uses_ref:
        raise ValueError(f'RLConfig.ref_model={rl_config.ref_model!r} is not used by rl_type={rl_type!r}'
                         f' with tuner={tuner!r}: CPO/ORPO fold the reference into their loss and LoRA '
                         'uses the adapter-disabled base as the reference, so no separate ref_model is loaded. '
                         'Remove it.')


# rl_types verified correct under the packed layout padding_free (and packing) produces: the forward
# unpacks each packed micro batch to per-sequence rows BEFORE calculate_loss (packing-agnostic).
_RL_PADDING_FREE_WIRED_TYPES = ('grpo', 'rft', 'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'gkd', 'rm')

_RL_PADDING_FREE_REFUSED_REASONS = {
    'ppo':
    "PPO's per-token value-head critic does not cover the packed / variable-length layout (run_ppo.py "
    'module docstring), so the critic would read a flattened row as one sequence.',
    'opsd':
    "OPSD's teacher->student per-token logps alignment across two independently-unpacked forwards is not "
    'verified under variable-length packing.',
    'mopd':
    "MOPD fuses K teacher logps channels onto the student frame; that K-way alignment across two "
    'independently-unpacked forwards is not verified under variable-length packing.',
}


def _check_rlhf_padding_free(template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
                            rl_config: Optional['RLConfig']) -> None:
    """Only some RLHF algorithms have a padding-free/packing training path.

    The forward unpacks packed batches before calculate_loss; refused types name the unsafe component.
    """
    if rl_config is None:
        return
    if not (template_config.padding_free or dataset_config.packing):
        return
    rl_type = getattr(rl_config, 'rl_type', None)
    if rl_type in _RL_PADDING_FREE_WIRED_TYPES:
        return
    feature = 'packing' if dataset_config.packing else 'padding_free'
    reason = _RL_PADDING_FREE_REFUSED_REASONS.get(rl_type)
    if reason is not None:
        raise ValueError(f'rl_type={rl_type!r} does not support {feature}: {reason} '
                         'Set the corresponding flag to False.')
    raise ValueError(f'rl_type={rl_type!r} does not support {feature}: only '
                     f'{"/".join(_RL_PADDING_FREE_WIRED_TYPES)} implement the variable-length training path it '
                     'produces. Set the corresponding flag to False.')


def _check_rlhf_padding_side(template_config: 'TemplateConfig', rl_config: Optional['RLConfig']) -> None:
    """RL logp extraction indexes each row's response positions and assumes right padding.

    Left padding silently mis-aligns every logp; dev refuses up front instead of corrupting the run.
    """
    if rl_config is None:
        return
    if template_config.padding_side != 'right':
        rl_type = getattr(rl_config, 'rl_type', None)
        raise ValueError(f'RL training (rl_type={rl_type!r}) requires padding_side="right", got '
                         f'{template_config.padding_side!r}: the logp extraction indexes each row\'s '
                         'response positions in a right-padded forward frame, so left padding would '
                         'silently mis-align every logp. Remove --padding_side left.')


# Online-RL rl_types whose Ulysses SP wiring is in place (run_grpo/run_rft call plan_sp_mesh):
# PPO's critic cannot follow the policy onto the SP mesh, and the offline family is not "under ray".
_RL_SP_WIRED_TYPES = ('grpo', 'rft')


def _check_sp_hf_feasibility(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                            dataset_config: 'DatasetConfig', distributed_config: 'DistributedConfig',
                            is_megatron: bool, sp: int) -> None:
    """Backend/layout feasibility guards shared by the SFT (local) and online-RL (ray) SP paths.

    Every branch raises rather than silently training with SP=1; only the world source differs.
    """
    if is_megatron:
        # TemplateConfig.sequence_parallel_size is the HF Ulysses knob; Megatron's sequence parallelism
        # is DistributedConfig.sequence_parallel (TP-SP), a different feature.
        raise ValueError(f'TemplateConfig.sequence_parallel_size={sp} only applies to the transformers backend, '
                         'but the active backend is megatron. Megatron sequence parallelism is '
                         'DistributedConfig.sequence_parallel (TP-SP over the tensor-parallel ranks) -- set that '
                         'instead, or switch DistributedConfig.backend.')

    if distributed_config.fsdp:
        raise NotImplementedError(f'sequence_parallel_size={sp} with DistributedConfig.fsdp is not supported yet: '
                                  'the FSDP x ulysses composition in twinkle is unvalidated. Use DDP/accelerate '
                                  '(the default strategy), or set sequence_parallel_size=1.')

    # Legacy asserts this at collate time (swift/template/base.py); dev fails fast at validation.
    if template_config.padding_side != 'right':
        raise ValueError(f'sequence_parallel_size={sp} requires padding_side="right" (got '
                         f'{template_config.padding_side!r}): the SP collator injects per-row position_ids that '
                         'assume right padding. legacy asserts the same at collate time; dev refuses up front.')

    if template_config.padding_free or dataset_config.packing:
        if model_config.attn_impl not in _SP_PADDING_FREE_ATTN_IMPLS:
            raise ValueError(f'sequence_parallel_size={sp} with padding_free requires a flash attention kernel: '
                             f'twinkle\'s SP strategy rejects the variable-length layout under '
                             f'attn_impl={model_config.attn_impl!r}. Use one of '
                             f'{", ".join(repr(i) for i in _SP_PADDING_FREE_ATTN_IMPLS)}, or set padding_free=False '
                             '(packing implies padding_free, so drop packing too).')


def _check_rlhf_sequence_parallel(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                                 dataset_config: 'DatasetConfig', distributed_config: 'DistributedConfig',
                                 rl_config: Optional['RLConfig'], is_megatron: bool) -> None:
    """Ulysses SP (TemplateConfig.sequence_parallel_size) for the online-RL recipes under ray.

    Only grpo/rft are wired; the world is nproc_per_node, NOT Platform.get_world_size().
    """
    if rl_config is None or template_config.sequence_parallel_size <= 1:
        return
    sp = template_config.sequence_parallel_size
    rl_type = getattr(rl_config, 'rl_type', None)
    if rl_type == 'ppo':
        raise ValueError(f'rl_type="ppo" does not support sequence_parallel_size={sp}: PPO\'s per-token '
                         'value-head critic runs without sequence-parallel / packed layouts, so a SP policy '
                         'and an SP=1 critic would slice_dp each forward_backward at a different '
                         'data_world_size and starve the critic\'s ranks. Set sequence_parallel_size=1 for PPO.')
    if rl_type not in _RL_SP_WIRED_TYPES:
        raise ValueError(f'rl_type={rl_type!r} does not support sequence_parallel_size={sp} on the dev path: '
                         f'only the online-RL recipes {"/".join(_RL_SP_WIRED_TYPES)} wire the Ulysses SP mesh '
                         '(plan_sp_mesh + a ray dp x ulysses device mesh). Set sequence_parallel_size=1.')

    _check_sp_hf_feasibility(model_config, template_config, dataset_config, distributed_config, is_megatron, sp)
    nproc = distributed_config.nproc_per_node
    if nproc is None or nproc < 2:
        raise ValueError(f'sequence_parallel_size={sp} requires DistributedConfig.nproc_per_node>=2 (got '
                         f'{nproc!r}): the ray \'model\' DeviceGroup needs at least two ranks to split a sequence '
                         'across. Set sequence_parallel_size=1 for a single-rank run.')
    if nproc % sp != 0:
        raise ValueError(f'nproc_per_node={nproc} is not divisible by sequence_parallel_size={sp}: the SP groups '
                         'must tile the ray ranks exactly. Adjust nproc_per_node or sequence_parallel_size.')


def _check_hf_sequence_parallel(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                                dataset_config: 'DatasetConfig', distributed_config: 'DistributedConfig',
                                is_megatron: bool, rl_config: Optional['RLConfig'] = None) -> None:
    """Guards for Ulysses sequence parallelism (TemplateConfig.sequence_parallel_size) on the HF SFT path.

    Every check raises; streaming is allowed; RLHF is delegated (early return below).
    """
    sp = template_config.sequence_parallel_size
    if sp <= 1:
        return
    if rl_config is not None:
        return

    _check_sp_hf_feasibility(model_config, template_config, dataset_config, distributed_config, is_megatron, sp)

    if distributed_config.mode != 'local':
        raise NotImplementedError(f'sequence_parallel_size={sp} is only wired for mode="local" (torchrun): under '
                                  "mode='ray' the model gets a pure data-parallel mesh (_apply_ray_placement), so "
                                  'SP would silently not apply. Run with torchrun, or set sequence_parallel_size=1.')

    # twinkle.initialize(mode='local') never calls dist.init_process_group -- world size comes from
    # Platform.get_world_size() (the WORLD_SIZE env torchrun sets), initialize's own source.
    from twinkle.utils import Platform
    world = Platform.get_world_size()
    if world < 2:
        raise ValueError(f'sequence_parallel_size={sp} requires torchrun (WORLD_SIZE>=2): a single process has no '
                         'ranks to split a sequence across. Set sequence_parallel_size=1 for a single-process run.')
    if world % sp != 0:
        raise ValueError(f'world_size={world} is not divisible by sequence_parallel_size={sp}: the SP groups must '
                         'tile the ranks exactly. Adjust the rank count or sequence_parallel_size.')

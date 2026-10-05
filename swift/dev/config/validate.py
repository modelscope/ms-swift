# Cross-config validation: the one place where rules spanning several Configs are enforced.

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        InferConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = logging.getLogger(__name__)


def validate_configs(
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    dataset_config: 'DatasetConfig',
    train_config: 'TrainConfig',
    distributed_config: 'DistributedConfig',
    checkpoint_config: Optional['CheckpointConfig'] = None,
    tuner_config: Optional['TunerConfig'] = None,
    rlhf_config: Optional['RLHFConfig'] = None,
    logging_config: Optional['LoggingConfig'] = None,
    quantize_config: Optional['QuantizeConfig'] = None,
    megatron_config: Optional['MegatronConfig'] = None,
    moe_config: Optional['MoEConfig'] = None,
    *,
    training: bool = True,
) -> None:
    """Validate constraints that span multiple Configs. Raises ValueError on an illegal combination.

    Call this BEFORE building anything heavy (dataset/model)
    """
    from swift.dev.builders.model import is_megatron_backend
    is_megatron = is_megatron_backend(distributed_config)

    _check_group_by_length(dataset_config, template_config)
    _check_lazy_tokenize(dataset_config)
    _check_data_sharding(dataset_config)
    _check_streaming(dataset_config, checkpoint_config)
    _check_backend_specific(model_config, dataset_config, train_config, distributed_config, is_megatron, tuner_config)
    _check_galore(train_config, tuner_config)
    _check_unsloth_strategy(tuner_config, distributed_config, template_config)
    _check_deepspeed_autotp(distributed_config)
    _check_eval_generation(train_config, template_config, distributed_config)
    _check_megatron_runtime_configs(megatron_config, moe_config, is_megatron)
    _check_megatron_optimizer(train_config, is_megatron)
    _check_muon(train_config, distributed_config, is_megatron)
    _check_megatron_recompute(train_config, distributed_config, is_megatron)
    _check_megatron_microbatch_schedule(train_config, distributed_config, is_megatron)
    _check_eval_iters(train_config)
    _check_megatron_attn_backend(model_config, template_config, is_megatron)
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
        rlhf_config,
        is_megatron,
        training=training)
    _check_save_total_limit(checkpoint_config, is_megatron)
    _check_logging(logging_config)
    _check_rlhf_ref_model(model_config, tuner_config, rlhf_config)
    _check_rlhf_advanced(train_config, rlhf_config)
    _check_rlhf_padding_free(template_config, dataset_config, rlhf_config)
    _check_rlhf_sequence_parallel(
        model_config, template_config, dataset_config, distributed_config, rlhf_config, is_megatron)
    # Packing-derived padding_free has already been resolved by process_configs. RLHF SP is fully governed
    # by _check_rlhf_sequence_parallel above, so this early-returns for a non-None rlhf_config.
    _check_hf_sequence_parallel(
        model_config, template_config, dataset_config, distributed_config, is_megatron, rlhf_config)


def _changed_fields(config) -> list:
    """Return user-visible fields that differ from defaults or were explicit on the CLI."""
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


def _check_rlhf_advanced(train_config: 'TrainConfig', rlhf_config: Optional['RLHFConfig']) -> None:
    """Validate online-RL features before models, teachers, or rollout workers are created."""
    if rlhf_config is None:
        return
    cfg = rlhf_config
    online_only = bool(cfg.chord_sft_dataset or cfg.advantage_reweight or cfg.sdar_loss_coef > 0 or cfg.dynamic_sample
                       or cfg.sync_ref_model or cfg.max_turns is not None)
    if online_only and cfg.rlhf_type != 'grpo':
        raise ValueError('CHORD, RLSD, SDAR, dynamic sampling, reference sync, and multi-turn are GRPO-only.')
    _check_dynamic_sampling(cfg)
    _check_grpo_controls(cfg)
    _check_reference_sync(cfg, train_config)
    _check_chord(cfg)
    _check_router_replay(cfg)
    _check_sampling_replay(cfg)
    _check_prm(cfg)
    _check_self_distillation(cfg, train_config)
    _check_distillation(cfg)
    _check_auxiliary_adapters(cfg)
    _check_multi_turn(cfg)
    _check_preference_reference(cfg)
    _check_dpo_loss(cfg)


#: The f-divergences DPO can stand in for the KL; ``f_divergence_type`` must name one of them.
_DPO_F_DIVERGENCE_TYPES = ('reverse_kl', 'forward_kl', 'js_divergence', 'alpha_divergence')


def _check_dpo_loss(cfg: 'RLHFConfig') -> None:
    """Validate the DPO loss surface: the f-divergence name and the multi-loss (MPO) weight alignment.

    Both are also enforced when the DPO loss is constructed, but checking here fails before any model is
    loaded and keeps the message command-neutral. Only applies to ``rlhf_type='dpo'`` -- the other
    preference losses (kto/simpo/cpo/orpo/rm) read neither knob.
    """
    if cfg.rlhf_type != 'dpo':
        return
    if cfg.f_divergence_type not in _DPO_F_DIVERGENCE_TYPES:
        raise ValueError(f'--f_divergence_type must be one of {list(_DPO_F_DIVERGENCE_TYPES)}, '
                         f'got {cfg.f_divergence_type!r}.')
    loss_types = cfg.loss_type or ['sigmoid']
    if cfg.loss_weights is not None and len(cfg.loss_weights) != len(loss_types):
        raise ValueError(f'--loss_weights must give one weight per --loss_type '
                         f'({len(cfg.loss_weights)} weights for {len(loss_types)} loss types).')


def _check_preference_reference(cfg: 'RLHFConfig') -> None:
    """Reject reference-free KTO: KTO has no reference-free form (W12).

    KTO's implicit reward IS the policy-vs-reference log-ratio, and its z_KL anchor is a KL against that
    same reference, so both terms need reference logps. ``reference_free=True`` would leave KTOLoss with
    none -- it raises mid-training -- so refuse here and point at the fix: under LoRA the adapter-disabled
    base is the reference (run_dpo builds it automatically), and full fine-tuning derives one from the
    policy init.
    """
    if cfg.rlhf_type == 'kto' and cfg.reference_free:
        raise ValueError('KTO cannot be reference-free: its implicit reward is the log-ratio against a '
                         'reference and z_KL is a KL against that reference. Remove --reference_free; '
                         'under LoRA the adapter-disabled base serves as the reference automatically, and '
                         'full fine-tuning derives one from the policy init.')


def validate_multi_turn_config(config: 'RLHFConfig') -> None:
    """Validate the shared multi-turn surface without imposing RL-training-only constraints."""
    _check_multi_turn(config)


def _check_multi_turn(cfg: 'RLHFConfig') -> None:
    # Multi-turn is on iff max_turns is set. Per-turn length is sampling_params.max_tokens;
    # max_trajectory_tokens caps the whole trajectory. Both are independent, either may be None.
    if cfg.max_turns is None:
        return
    if cfg.max_turns < 1:
        raise ValueError('max_turns must be >= 1.')
    if cfg.max_trajectory_tokens is not None and cfg.max_trajectory_tokens < 1:
        raise ValueError('max_trajectory_tokens must be >= 1.')


def validate_rollout_config(rollout_config: Optional['RolloutConfig'],
                            multi_turn_config: Optional['RLHFConfig'],
                            rlhf_config: Optional['RLHFConfig'] = None,
                            tuner_config: Optional['TunerConfig'] = None) -> None:
    """Validate the rollout tools/sandbox surface without imposing RL-training-only constraints.

    ``multi_turn_config`` is whatever carries ``max_turns`` for the active recipe (the RLHFConfig for
    GRPO, the reward/multi-turn config for sampling); it may be None when multi-turn is not offered.
    ``rlhf_config`` is the training config when this rollout backs an RL recipe (None for sampling); it
    is only read by :func:`_check_async_mode`, which cross-checks the ``async_mode`` rollout knob against
    the training surface. ``tuner_config`` likewise is only read there, to tell a full-parameter run from
    an adapter run (``weight_sync_strategy='adapter_snapshot'`` needs an adapter to save). Both checks
    no-op when ``async_mode='none'``, so sampling recipes -- which never set it -- are unaffected.
    """
    _check_rollout_sampler(rollout_config)
    _check_rollout_tools(rollout_config, multi_turn_config)
    _check_offload_knobs(rollout_config)
    _check_async_mode(rollout_config, rlhf_config, tuner_config)


#: The online loops that roll out per prompt batch and train on the collected samples, so they can overlap
#: generation of batch ``N+1`` with training of batch ``N`` (:func:`overlap_rollout_batches`). RFT is
#: excluded (it bootstraps a fixed sample set with a plain cross-entropy objective and no off-policy
#: correction, and its ``fit`` iterates bootstrap rounds rather than prompt batches); the offline preference
#: family (dpo/kto/cpo/orpo/simpo/rm) never rolls out, so it is excluded too.
_ASYNC_RLHF_TYPES = frozenset({'grpo', 'ppo', 'gkd', 'opsd', 'mopd'})


#: Knobs read ONLY by the per-sample streaming driver (GRPO/PPO under ``one_step_off``/``fully_async``),
#: with the value that is inert (ignored) everywhere else -- the synchronous driver (any recipe) and the
#: distillation loops' single-batch overlap (``overlap_rollout_batches``). An explicit non-default value
#: there would be a silent dead knob, so it is rejected rather than ignored. ``allow_partial_rollout``
#: joins them: interrupting and resuming an in-flight generation is only meaningful when a streaming
#: publish overwrites the sampler's live weights under it (``in_place``); the synchronous driver syncs at
#: the sampler idle point, so there is nothing to resume.
_FULLY_ASYNC_ONLY_DEFAULTS = {
    'max_staleness': 1,
    'weight_sync_strategy': 'adapter_snapshot',
    'allow_partial_rollout': False,
    'parameter_sync_step': 1
}


def _check_async_mode(rollout_config: Optional['RolloutConfig'], rlhf_config: Optional['RLHFConfig'],
                      tuner_config: Optional['TunerConfig'] = None) -> None:
    """Reject ``async_mode`` combinations the online loops cannot honour (fail-loudly).

    ``async_mode`` picks how far the rollout may run ahead of training, i.e. how off-policy the trained
    batch is. ``'none'`` is synchronous; ``'one_step_off'`` overlaps ONE batch (driver-side double buffer,
    staleness pinned to 1 because the shared-GPU weight sync needs the sampler idle every step);
    ``'fully_async'`` lets a disaggregated sampler run several versions ahead (staleness up to
    ``max_staleness``). Every overlapping regime is a real off-policy gap and a real deployment
    constraint, so each of the following is refused here rather than silently degrading:

    - Only the loops in :data:`_ASYNC_RLHF_TYPES` implement the overlapped dispatch; RFT and the offline
      family override ``fit`` (or never roll out) and would ignore the flag. GRPO and PPO run BOTH
      overlapping regimes through the per-sample streaming driver (``grpo_async.StreamingGRPOLoop`` /
      ``ppo_async.StreamingPPOLoop``, both composed from ``recipe._streaming_loop.StreamingLoopMixin``),
      differing only in the staleness knob; ``'fully_async'`` narrows to GRPO and PPO because the
      distillation loops still overlap a single batch (``overlap_rollout_batches``) and have no
      deep-buffer driver.
    - GRPO trains on raw sampled tokens, so any staleness needs the token-level
      ``rollout_importance_sampling_mode`` correction. PPO's clipped surrogate and the distillation
      family's teacher target already bound the update, so they do not require it.
    - Weight sync rewrites live sampler weights and needs the sampler on its own GPUs to overlap
      generation with training, so ``vllm_mode='disaggregated'`` is mandatory; ``'colocate'`` time-shares
      one DeviceGroup and cannot overlap.
    - ``dynamic_sample`` regenerates zero-variance groups adaptively after seeing rewards, so it cannot be
      pre-submitted ahead of training. Multi-turn CAN now overlap: GRPO/PPO admit each episode on the
      per-sample streaming driver (a whole episode runs turn-by-turn on its own thread), but the
      distillation family still overlaps a single batch through ``submit_generate`` -- which drives only the
      single-turn sampler -- so ``max_turns`` is refused there until their streaming migration.
    - Async distillation must be purely on-policy (``lmbda==1.0``): an off-policy dataset round generates
      nothing, so there is no batch to admit ahead.
    """
    if rollout_config is None:
        return
    # allow_partial_rollout is a real sampler capability now (twinkle's PartialRolloutMixin: the dev vLLM/SGLang
    # sampler can interrupt an in-flight generation and resume it on freshly-synced weights), but it is only
    # MEANINGFUL under the streaming driver's in_place publication, where a publish overwrites the sampler's
    # single live weight copy under an in-flight generation. Under 'none' (and the distillation loops'
    # one_step_off) it is inert (the sync happens at the sampler idle point, nothing is in flight to resume)
    # and _reject_fully_async_only_knobs refuses an explicit True; for GRPO/PPO one_step_off/fully_async,
    # _check_streaming_publication validates it per weight_sync_strategy (required for in_place, inert for
    # adapter_snapshot). So there is no blanket rejection here -- only the per-regime checks below.
    mode = rollout_config.async_mode
    if mode == 'none':
        _reject_fully_async_only_knobs(rollout_config, mode)
        return
    # The legacy bool is the old spelling of 'one_step_off'; process._derive_async_mode folds it in when
    # async_mode is unset, so reaching here with async_generate True AND a different explicit mode is an
    # ambiguous request (both were set), not something to silently resolve.
    if rollout_config.async_generate and mode != 'one_step_off':
        raise ValueError(
            f'--async_generate is the legacy spelling of --async_mode one_step_off, but --async_mode '
            f'{mode!r} was also set. Drop --async_generate and express the regime with --async_mode alone.')
    flag = '--async_generate' if rollout_config.async_generate else f'--async_mode {mode}'
    rlhf_type = None if rlhf_config is None else rlhf_config.rlhf_type
    if rlhf_type not in _ASYNC_RLHF_TYPES:
        raise ValueError(
            f'{flag} is only implemented for the online loops that roll out per prompt batch and '
            f'train on the collected samples (grpo/ppo/gkd/opsd/mopd); got rlhf_type={rlhf_type!r}. The '
            'other recipes override the training loop (or never roll out) and would silently ignore it. '
            'Remove the async setting.')
    if rlhf_type == 'grpo' and rlhf_config.rollout_importance_sampling_mode is None:
        raise ValueError(
            f'{flag} trains GRPO on a rollout batch that is stale (off-policy), so it requires off-policy '
            'correction: set --rollout_importance_sampling_mode (e.g. token_truncate or sequence_mask). '
            'Use --async_mode none if you want a strictly on-policy loop.')
    if rollout_config.vllm_mode != 'disaggregated':
        raise ValueError(
            f'{flag} needs the sampler on its own GPUs to overlap generation with training, but '
            f"vllm_mode={rollout_config.vllm_mode!r}. Set --vllm_mode disaggregated; 'colocate' time-shares "
            'one DeviceGroup and cannot overlap.')
    if rlhf_config.dynamic_sample:
        raise ValueError(
            f'{flag} cannot be combined with --dynamic_sample: dynamic sampling regenerates '
            'zero-variance groups after rewards are seen, which cannot be pre-submitted ahead of training. '
            'Disable one of them.')
    if rlhf_config.advantage_estimator == 'remax':
        raise ValueError(
            f'{flag} cannot be combined with --advantage_estimator remax: ReMax scores a greedy '
            '(argmax) baseline completion per prompt AFTER the sampled rollout, and that extra synchronous '
            'generation pass cannot be pre-submitted ahead. Remove the async setting or pick another '
            'advantage_estimator.')
    if rlhf_config.max_turns is not None and rlhf_type in {'gkd', 'opsd', 'mopd'}:
        raise ValueError(
            f'{flag} cannot be combined with a multi-turn rollout (max_turns is set) for '
            f'rlhf_type={rlhf_type!r}: the distillation loops still overlap a single batch through '
            'submit_generate, which drives only the single-turn sampler and has no per-episode multi-turn '
            'admission. GRPO/PPO run multi-turn through the per-sample streaming driver (each episode is '
            'admitted on its own thread). Remove the async setting or unset --max_turns.')
    if rlhf_type in {'gkd', 'opsd', 'mopd'} and rlhf_config.lmbda != 1.0:
        raise ValueError(
            f'{flag} overlaps generation with training by pre-submitting the next prompt batch, '
            f'which only purely on-policy distillation can do; rlhf_type={rlhf_type!r} with '
            f'--lmbda={rlhf_config.lmbda} mixes in off-policy dataset rounds that generate nothing and '
            'cannot be pre-submitted. Set --lmbda 1 for on-policy distillation, or remove the async setting.')
    if rlhf_type in ('grpo', 'ppo'):
        # GRPO/PPO run BOTH overlapping regimes through the per-sample StreamingDriver, so the
        # publication knobs are live under one_step_off as well as fully_async.
        _check_streaming_publication(rollout_config, rlhf_config, tuner_config, mode)
    elif mode == 'one_step_off':
        # The distillation loops still overlap a single batch (overlap_rollout_batches); their streaming
        # migration is a later phase, so the deep-buffer publication knobs stay inert here and an
        # explicit non-default value would silently do nothing.
        _reject_fully_async_only_knobs(rollout_config, mode)
    else:
        raise ValueError(
            f'{flag} with --async_mode fully_async is only implemented for GRPO and PPO (the loops with '
            f'the per-sample streaming, sparsely-published driver); got rlhf_type={rlhf_type!r}. The '
            'distillation loops overlap a single batch -- use --async_mode one_step_off for them.')


def _reject_fully_async_only_knobs(rollout_config: 'RolloutConfig', mode: str) -> None:
    """Refuse a streaming-only knob explicitly set to a non-inert value where no streaming driver runs.

    ``max_staleness``/``weight_sync_strategy``/``allow_partial_rollout``/``parameter_sync_step`` steer only
    the per-sample streaming driver (GRPO/PPO under ``one_step_off``/``fully_async``); the synchronous
    driver and the distillation loops' single-batch overlap never read them, so an explicit non-default
    value would silently do nothing. Fail loudly instead (``_changed_fields`` limits this to values the
    user actually set).
    """
    changed = _changed_fields(rollout_config)
    for name, inert in _FULLY_ASYNC_ONLY_DEFAULTS.items():
        if name in changed and getattr(rollout_config, name) != inert:
            raise ValueError(
                f'--{name}={getattr(rollout_config, name)!r} only governs the per-sample streaming driver '
                f'(grpo/ppo under --async_mode one_step_off/fully_async); the driver selected here '
                f'(--async_mode {mode}) never reads it (its inert value is {inert!r}), so it would silently '
                'do nothing. Use a streaming regime, or drop it.')


def _check_streaming_publication(rollout_config: 'RolloutConfig', rlhf_config: 'RLHFConfig',
                                 tuner_config: Optional['TunerConfig'], mode: str) -> None:
    """Guards for the per-sample streaming driver (GRPO/PPO under ``one_step_off`` and ``fully_async``).

    Both overlapping regimes run the SAME per-sample ``StreamingDriver``, differing only in the staleness
    knob, so a publication always happens while the sampler has generations in flight and must not corrupt
    them. Three things are checked (the strategy pairing mirrored at construction by
    ``recipe._streaming_loop.StreamingLoopMixin._check_streaming_config``, so a loop built directly is
    guarded too):

    * ``parameter_sync_step`` is the publish cadence -- one weight publication every K recorded steps; it
      must be >= 1 (0 would never publish a trained policy).
    * staleness: ``one_step_off`` pins the driver to a single lookahead window (staleness 1), so an
      explicit non-1 ``max_staleness`` there is ambiguous and refused (use ``fully_async`` for a deeper
      buffer); ``fully_async`` allows any ``max_staleness >= 1``.
    * the publish mechanism, per ``weight_sync_strategy``:

      - ``adapter_snapshot`` pins each trained version as its own LoRA adapter path, so publishing writes a
        NEW path and never disturbs a running generation. It needs an adapter run (a full-parameter policy
        has no adapter to pin) and must NOT set ``allow_partial_rollout`` -- nothing is overwritten in
        place, so there is no generation to interrupt and resume and the flag would be inert.
      - ``in_place`` overwrites the sampler's single live weight copy, so it is sound ONLY with partial
        rollout: the publish aborts every in-flight generation and each resumes from its own tokens on the
        fresh weights (twinkle ``PartialRolloutMixin`` + ``InPlaceWeightSync`` abort-on-publish), so no
        generation decodes across the update and its logprobs stay correctable. It supports a
        full-parameter policy (merged base weights over NCCL when disaggregated, CUDA IPC when colocated)
        as well as LoRA, and therefore REQUIRES ``allow_partial_rollout``.
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
    elif rollout_config.max_staleness < 1:
        raise ValueError(
            f'--max_staleness must be >= 1 under --async_mode fully_async (0 would admit nothing ahead of '
            f'the oldest untrained window, i.e. the synchronous loop); got {rollout_config.max_staleness}.')
    strategy = rollout_config.weight_sync_strategy
    tuner = getattr(tuner_config, 'tuner', 'full') if tuner_config is not None else 'full'
    if strategy == 'adapter_snapshot':
        # Per-version LoRA pinning needs an adapter to pin; a full-parameter policy has none. The sound
        # full-parameter deep buffer is in_place + partial rollout (below), so point the user there.
        if tuner == 'full':
            raise ValueError(
                f"--async_mode {mode} with --weight_sync_strategy adapter_snapshot pins each trained "
                "version as a LoRA adapter path, but this is a full-parameter run (tuner='full') with no "
                'adapter to pin. Configure an adapter-based tuner (e.g. LoRA), or use --weight_sync_strategy '
                'in_place with --allow_partial_rollout, which overwrites the sampler\'s live weights and '
                'resumes each in-flight generation on them (the full-parameter deep buffer).')
        if rollout_config.allow_partial_rollout:
            raise ValueError(
                '--allow_partial_rollout is inert under --weight_sync_strategy adapter_snapshot: each version '
                'is pinned as its own adapter path, so publishing never overwrites a weight copy an in-flight '
                'generation decodes and there is nothing to interrupt and resume. Drop --allow_partial_rollout, '
                'or use --weight_sync_strategy in_place (which overwrites one live copy and so needs it).')
    elif strategy == 'in_place':
        # Overwriting the single live copy under an in-flight generation is unsound unless each generation is
        # aborted and resumed on the fresh weights, which is exactly what allow_partial_rollout enables.
        if not rollout_config.allow_partial_rollout:
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
    """Warn on offload/sleep knobs dev cannot honour exactly as set (fail-loudly, non-fatal).

    In the colocate placement the rollout offload is *structural and atomic*: twinkle's ``offload_to_cpu``
    bundles the model and the optimizer together and is coupled to the wake/sleep memory handover in
    ``ColocateHandover.enter()``, so it always runs and can be neither switched off nor split into two
    independent tiers. ``offload_model``/``offload_optimizer`` therefore stay as flags for CLI parity but
    do not gate anything, and any explicit request that deviates from "both on" is warned about rather
    than silently applied.

    In the disaggregated (or server) placement the sampler owns its own GPUs, so offloading the trainer's
    model/optimizer to make room for it is moot, and ``sleep_level`` is inert because the handover never
    sleeps a sampler that is not sharing the device.

    Only explicitly-set values are flagged (``_changed_fields``), so defaults stay silent.
    """
    if rollout_config is None:
        return
    changed = _changed_fields(rollout_config)
    model_set = 'offload_model' in changed
    optim_set = 'offload_optimizer' in changed
    offload_model = rollout_config.offload_model
    offload_optimizer = rollout_config.offload_optimizer
    if rollout_config.vllm_mode == 'colocate':
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
                'vllm_mode=%r places the sampler on its own GPUs, so offloading the trainer model/optimizer '
                'to make room for it does not apply: ignoring --offload_model/--offload_optimizer.',
                rollout_config.vllm_mode)
        if 'sleep_level' in changed:
            logger.warning(
                'vllm_mode=%r does not share GPUs between trainer and sampler, so the sampler is never put '
                'to sleep between steps: --sleep_level is inert here (it only takes effect under '
                "--vllm_mode colocate).", rollout_config.vllm_mode)


def _check_rollout_sampler(rollout_config: Optional['RolloutConfig']) -> None:
    """Reject a rollout engine dev cannot wire as an independent-process, weight-synced sampler.

    ``'vllm'`` and ``'sglang'`` are both wired (GRPO/PPO/RFT and the distill family read it for the
    rollout sampler). ``'transformers'`` is not: the online loop syncs the trained policy into the
    sampler every step through a CheckpointEngineManager, which needs a CheckpointEngineMixin engine,
    and TransformersSampler deliberately has none (it generates on the trainer's own weights). The
    ``rollout_sampler`` Literal already excludes it; this is the loud backstop so a mis-set value fails
    at validation rather than silently running without weight sync.
    """
    if rollout_config is None:
        return
    if rollout_config.rollout_sampler not in ('vllm', 'sglang'):
        raise ValueError(
            f"rollout_sampler={rollout_config.rollout_sampler!r} cannot back an online-RL rollout: weight "
            "sync into the sampler needs a CheckpointEngineMixin engine, which only 'vllm' and 'sglang' "
            "provide (TransformersSampler has none by design). Use rollout_sampler='vllm' or 'sglang'.")


def _check_rollout_tools(rollout_config: Optional['RolloutConfig'],
                         multi_turn_config: Optional['RLHFConfig']) -> None:
    if rollout_config is None:
        return
    if rollout_config.sandbox_num_envs < 1:
        raise ValueError(f'sandbox_num_envs must be >= 1, got {rollout_config.sandbox_num_envs}.')
    if not rollout_config.tools:
        return
    # Tools are injected per-turn by the multi-turn engine, so a single-turn rollout has no turn to
    # call them in -- requiring max_turns turns a silent no-op into a clear error.
    max_turns = multi_turn_config.max_turns if multi_turn_config is not None else None
    if max_turns is None:
        raise ValueError(
            'RolloutConfig.tools requires a multi-turn rollout: set max_turns (>= 1) so the engine has '
            'turns in which to call the tools. A single-turn rollout cannot invoke them.')


def validate_infer_config(infer_config: Optional['InferConfig']) -> None:
    """Validate the best-of-n synthesis surface without imposing RL-training-only constraints.

    Synthesis runs through the ``infer`` CLI rather than a training recipe, so this is kept apart from
    :func:`validate_configs`; it only rejects combinations that cannot work at generation time.
    """
    if infer_config is None:
        return
    # save_rollout_tokens persists the per-token feature the rollout produced; the message-only ``client``
    # teacher exposes no token IDs or logprobs, so there would be nothing to write. Reject the pairing up
    # front instead of silently emitting rows that carry no token path.
    if infer_config.save_rollout_tokens and infer_config.sampler == 'client':
        raise ValueError(
            "save_rollout_tokens requires a token-capable local backend, but sampler='client' is "
            'message-only (no token IDs or logprobs). Use a local backend (transformers/vllm/sglang) or '
            'drop --save_rollout_tokens.')
    # store_format='arrow' serialises the whole run once at the end and writes no checkpoint files, so it
    # cannot continue a previous run -- resume needs the jsonl checkpoint writer's .tmp/.resume/state.
    if infer_config.store_format == 'arrow' and infer_config.resume:
        raise ValueError("store_format='arrow' writes one Arrow table at the end of the run and cannot "
                         'checkpoint, so it cannot be combined with resume=True. Use --store_format jsonl '
                         'for a resumable run.')


#: The GRPO-family ``loss_type`` values dev accepts (each maps onto a twinkle ``GRPOLoss`` subclass via
#: ``torch_loss_mapping``). Kept as one constant so the GRPO-only gate in :func:`_check_grpo_controls` and
#: the support check in :func:`_check_grpo_loss_type` cannot drift apart. The verl-ported variants
#: (gpg / dro / dppo_tv / dppo_kl / geo_mean / clip_cov / kl_cov) join the pre-existing set.
_GRPO_LOSS_TYPES = frozenset({
    'grpo', 'dapo', 'fipo', 'gspo', 'sapo', 'cispo', 'bnpo', 'dr_grpo', 'real', 'gpg', 'dro', 'dppo_tv',
    'dppo_kl', 'geo_mean', 'clip_cov', 'kl_cov'
})


def _check_grpo_controls(cfg: 'RLHFConfig') -> None:
    grpo_only = bool(
        cfg.log_completions or cfg.num_iterations != 1 or cfg.delta is not None
        or cfg.importance_sampling_level != 'token' or cfg.overlong_filter or cfg.log_entropy
        or cfg.top_entropy_quantile != 1.0 or cfg.rollout_importance_sampling_mode
        or cfg.log_rollout_offpolicy_metrics or cfg.off_policy_sequence_mask_delta is not None
        or loss_type in _GRPO_LOSS_TYPES)
    if cfg.rlhf_type != 'grpo':
        if grpo_only:
            raise ValueError('GRPO clipping, replay, entropy, FIPO, and completion logging controls are GRPO-only.')
        return
    _check_grpo_loss_type(cfg, loss_type)
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


def _check_grpo_loss_type(cfg: 'RLHFConfig', loss_type: str) -> None:
    supported = _GRPO_LOSS_TYPES
    if len(cfg.loss_type or []) > 1:
        raise ValueError('GRPO supports exactly one loss_type.')
    if loss_type not in supported:
        raise ValueError(f'Unsupported GRPO loss_type={loss_type!r}; expected one of {sorted(supported)}.')
    # GSPO IS sequence-level importance sampling; an explicit token-level request contradicts it and would
    # silently downgrade GSPO to token-level GRPO (W13). The default 'token' is promoted to 'sequence' by
    # configure_rlhf_loss, so only an explicitly-set 'token' is a real contradiction worth rejecting.
    if (loss_type == 'gspo' and cfg.importance_sampling_level == 'token'
            and 'importance_sampling_level' in _changed_fields(cfg)):
        raise ValueError(
            'loss_type=gspo IS sequence-level importance sampling; --importance_sampling_level token contradicts '
            'it and would silently downgrade GSPO to token-level GRPO. Drop the flag (gspo defaults to '
            'sequence-level), or use loss_type=grpo for token-level IS.')
    _check_grpo_loss_knobs(cfg, loss_type)


def _check_grpo_loss_knobs(cfg: 'RLHFConfig', loss_type: str) -> None:
    """Fail-loudly on the verl-ported loss variants' hyperparameters before training starts.

    The twinkle loss constructors validate these too, but checking at config time surfaces a bad value in
    the argument pass rather than deep in the first training step. Each knob is read by exactly one variant
    (see ``_online_loss_kwargs``), so only the matching variant's knob is checked.
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


def _check_dynamic_sampling(cfg: 'RLHFConfig') -> None:
    if not cfg.dynamic_sample:
        return
    if not (cfg.orm or cfg.reward_model):
        raise ValueError(
            'dynamic_sample requires orm or reward_model because it filters groups by reward variance.')
    if cfg.num_generations < 2:
        raise ValueError('dynamic_sample requires num_generations >= 2 to measure reward variance.')
    if cfg.max_resample_times < 1:
        raise ValueError('max_resample_times must be >= 1 when dynamic_sample is enabled.')


def _check_reference_sync(cfg: 'RLHFConfig', train_config: 'TrainConfig') -> None:
    if not cfg.sync_ref_model:
        return
    if cfg.beta in (None, 0, 0.0):
        raise ValueError('sync_ref_model requires beta > 0 and an active reference model.')
    if cfg.ref_model_sync_steps < 1:
        raise ValueError('ref_model_sync_steps must be >= 1.')
    if not 0.0 <= cfg.ref_model_mixup_alpha <= 1.0:
        raise ValueError('ref_model_mixup_alpha must be in [0, 1].')
    del train_config


def _check_chord(cfg: 'RLHFConfig') -> None:
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


def _check_router_replay(cfg: 'RLHFConfig') -> None:
    """Reject MoE routing-replay combinations dev cannot honour, so the knob is never silently dropped.

    ``router_replay_mode`` was a parsed-but-unwired dead knob; R2/R3 are now wired into GRPOLoop. Two
    combinations remain infeasible and must fail loudly here rather than corrupt a run:

    - Only GRPO consumes it: the RECORD/REPLAY driving lives in ``GRPOLoop._mini_batch_kwargs``, while the
      RFT/GKD/OPSD/PPO loops override ``fit`` and build their own ``forward_backward`` kwargs, so a non-GRPO
      run would set ``enable_router_replay`` on the model yet never pass a replay action -- a silent no-op.
    - CHORD appends auxiliary SFT rows that carry no expert routing, so they cannot share the REPLAY forward
      (twinkle's ``set_router_replay_data`` aligns routing to every row); the two are mutually exclusive.

    The MoE-only and sampler-export requirements are checked at runtime instead (a dense policy records no
    routing; a non-exporting engine leaves R3 with none) -- both surface as loud errors in the loop, since
    validate has no model/engine to inspect.
    """
    mode = cfg.router_replay_mode
    if mode == 'disabled':
        return
    if cfg.rlhf_type != 'grpo':
        raise ValueError(f'router_replay_mode={mode!r} is GRPO-only: R2/R3 routing replay is driven by '
                         'GRPOLoop, and the RFT/GKD/OPSD/PPO loops build their own forward_backward kwargs '
                         'without it. Use rlhf_type=grpo, or router_replay_mode=disabled.')
    if cfg.chord_sft_dataset:
        raise ValueError('router_replay_mode cannot be combined with CHORD: the auxiliary SFT rows carry no '
                         'expert routing, so they cannot share the REPLAY forward. Drop chord_sft_dataset, or '
                         'set router_replay_mode=disabled.')


def _check_sampling_replay(cfg: 'RLHFConfig') -> None:
    """Reject sampling-replay combinations dev cannot honour, so the knob is never silently dropped.

    ``enable_sampling_replay`` makes the training forward recompute each token's log-prob over the sampler's
    exact support set (twinkle ``replayed_selective_log_softmax``), so the GRPO ratio is measured against the
    distribution the token was actually drawn from. twinkle's GRPOLoss/forward impose hard constraints; the
    ones decidable from RLHFConfig alone are enforced here, the rest fail loudly at their own site:

    - GRPO-only: the ``sampling_masks`` column is assembled by ``GRPOLoop`` (``_rollout_step`` /
      ``_mini_batch_kwargs``); the RFT/GKD/OPSD/PPO loops override ``fit`` and never carry it.
    - No loss-side KL penalty: GRPOLoss raises when ``enable_sampling_replay`` and ``beta != 0``. dev only
      forwards ``beta`` to the loss when it is set, KL is not folded into the reward, and KL is calculated
      (mirroring ``_online_loss_kwargs``) -- so only that resolved value is checked here; a KL-in-reward run
      keeps the loss ``beta`` at 0 and is allowed.
    - No CHORD: its auxiliary SFT rows are not sampled, so they have no sampling mask to replay against
      (GRPOLoss rejects the resulting ``loss_mask != trainable``); the two are mutually exclusive.

    The entropy bonus (``entropy_coef``) has no dev knob, so it stays 0 and needs no guard. Sequence/context
    parallelism is rejected by the twinkle forward itself (and, today, by the blanket RLHF-SP guard), which
    this config-only check has no parallel sizes to inspect.
    """
    if not cfg.enable_sampling_replay:
        return
    if cfg.rlhf_type != 'grpo':
        raise ValueError(f'enable_sampling_replay is GRPO-only: the sampling_masks column is assembled by '
                         f'GRPOLoop, but rlhf_type={cfg.rlhf_type!r} builds its own forward_backward kwargs '
                         'without it. Use rlhf_type=grpo, or disable enable_sampling_replay.')
    calculate_kl = cfg.calculate_KL is not False
    loss_beta = cfg.beta if (cfg.beta is not None and not cfg.kl_in_reward and calculate_kl) else None
    if loss_beta:
        raise ValueError(f'enable_sampling_replay forbids a GRPO KL penalty in the loss (beta must be 0), but '
                         f'beta={loss_beta} would reach GRPOLoss. Set beta=0, or fold the KL into the reward '
                         'with kl_in_reward=True, or disable enable_sampling_replay.')
    if cfg.chord_sft_dataset:
        raise ValueError('enable_sampling_replay cannot be combined with CHORD: the auxiliary SFT rows are not '
                         'sampled, so they have no sampling mask to replay against. Drop chord_sft_dataset, or '
                         'disable enable_sampling_replay.')
    if cfg.max_turns is not None:
        raise ValueError('enable_sampling_replay does not support multi-turn rollouts: tool/observation turns '
                         'are trainable but were not drawn from the sampled policy, so they have no sampling '
                         'mask to replay against (GRPOLoss rejects the resulting loss_mask != trainable). Set '
                         'max_turns=None, or disable enable_sampling_replay.')


def _check_prm(cfg: 'RLHFConfig') -> None:
    """Reject GRPO process-reward (PRM) combinations the loop cannot honour, so no knob is silently dead.

    ``--prm`` is a dual-use selector: best-of-n synthesis ranks with it as a scalar axis, while GRPO turns it
    into a per-token *process* reward (each reasoning step scored by its response prefix, broadcast onto the
    step's tokens -- see :mod:`swift.dev.rewards.prm`). Only the GRPO consumer is validated here; for any other
    ``rlhf_type`` (synthesis runs under the ``dpo`` default) ``prm`` is just a ranking axis and the
    process-reward knobs do not apply.

    - Multi-turn is unsupported: a step prefix would span several assistant turns, so per-step scoring is
      undefined (``GRPOLoop._apply_prm`` raises at runtime; refused here up front).
    - Mutually exclusive with RLSD/SDAR: both replace the scalar advantage with a per-token one from a
      different source (teacher-signal reweighting vs process reward), and ``_training_advantage`` passes a
      per-token PRM advantage straight through -- it cannot also apply the teacher reweighting.
    - A process-reward knob with no ``--prm`` channel is dead: ``prm_scorer``/``prm_step_delimiter`` only shape
      how PRM steps are segmented and scored, so setting one without any PRM item would do nothing.
    """
    if cfg.rlhf_type != 'grpo':
        return
    scorer_knobs = sorted({'prm_scorer', 'prm_step_delimiter'}.intersection(_changed_fields(cfg)))
    if not cfg.prm:
        if scorer_knobs:
            details = ', '.join(f'--{name}' for name in scorer_knobs)
            raise ValueError(
                f'{details} shape the GRPO process reward but no --prm channel is set, so they would do '
                'nothing. Add a --prm item (a rule name or a frozen PRM model id), or drop them.')
        return
    if cfg.max_turns is not None:
        raise ValueError(
            'The --prm process reward does not support multi-turn rollouts: a reasoning-step prefix would span '
            'several assistant turns, so per-step scoring is undefined. Set max_turns=None, or drop --prm.')
    if cfg.advantage_reweight == 'rlsd' or cfg.sdar_loss_coef > 0:
        raise ValueError(
            'The --prm process reward cannot be combined with RLSD/SDAR: both turn the scalar advantage into a '
            'per-token one from different sources (process reward vs teacher-signal reweighting), and only one '
            'can drive the loss. Drop --prm, or disable advantage_reweight=rlsd / sdar_loss_coef.')


def _check_self_distillation(cfg: 'RLHFConfig', train_config: 'TrainConfig') -> None:
    _check_rlsd(cfg)
    _check_sdar(cfg)
    if (cfg.advantage_reweight == 'rlsd' or cfg.sdar_loss_coef > 0) and train_config.use_liger_kernel:
        raise ValueError('RLSD and SDAR require the unfused per-token loss path; disable use_liger_kernel.')


def _check_rlsd(cfg: 'RLHFConfig') -> None:
    if cfg.advantage_reweight != 'rlsd':
        return
    if not 0.0 <= cfg.rlsd_lambda <= 1.0:
        raise ValueError('rlsd_lambda must be in [0, 1].')
    if cfg.rlsd_reweight_clip_range < 0:
        raise ValueError('rlsd_reweight_clip_range must be >= 0.')
    if cfg.rlsd_lambda_warmup_steps < 0 or cfg.rlsd_lambda_decay_steps < 0:
        raise ValueError('RLSD warmup and decay steps must be non-negative.')
    if not (cfg.orm or cfg.reward_model):
        raise ValueError('advantage_reweight=rlsd requires orm or reward_model.')


def _check_sdar(cfg: 'RLHFConfig') -> None:
    if cfg.sdar_loss_coef <= 0:
        return
    if cfg.sdar_gate_beta <= 0:
        raise ValueError('sdar_gate_beta must be > 0.')
    if cfg.advantage_reweight == 'rlsd':
        raise ValueError('SDAR and RLSD cannot be enabled together.')


def _check_distillation(cfg: 'RLHFConfig') -> None:
    """Validate the on-policy distillation family (gkd / opsd / mopd).

    ``lmbda`` and the "offload / teacher_deepspeed need a distinct local teacher_model" guards are shared
    by all three (each may generate on-policy and may distil from a separate frozen teacher). The
    full-vocab JSD knobs (``temperature`` / ``gkd_logits_topk`` / ``sft_alpha``) are GKD-only: OPSD/MOPD
    use the logits-free sampled-token k3 surrogate, which reads none of them.
    """
    if cfg.rlhf_type not in ('gkd', 'opsd', 'mopd'):
        return
    if not 0.0 <= cfg.lmbda <= 1.0:
        raise ValueError(f'{cfg.rlhf_type} lmbda must be in [0, 1].')
    if cfg.offload_teacher_model and cfg.teacher_model is None:
        raise ValueError('offload_teacher_model requires a distinct local teacher_model.')
    if cfg.teacher_deepspeed and cfg.teacher_model is None:
        raise ValueError('teacher_deepspeed requires a distinct local teacher_model.')
    if cfg.rlhf_type in ('opsd', 'mopd'):
        # OPSD/MOPD v1 distils with the sampled-token k3 surrogate only: it applies no reference-KL term,
        # so a non-zero beta would be a silently-ignored knob. Reject rather than drop. (The GKD-only
        # full-vocab knobs temperature/gkd_logits_topk/sft_alpha are simply never read here.)
        if cfg.beta not in (None, 0, 0.0):
            raise ValueError(f'{cfg.rlhf_type} v1 has no reference-KL term; beta={cfg.beta!r} would be ignored. '
                             'Leave beta unset/0 -- self-distillation pulls the student toward the teacher directly.')
        return
    if cfg.rlhf_type != 'gkd':
        return
    if cfg.sft_alpha < 0:
        raise ValueError('GKD sft_alpha must be >= 0.')
    if cfg.temperature <= 0:
        raise ValueError('GKD temperature must be > 0.')
    if cfg.gkd_logits_topk is not None and cfg.gkd_logits_topk < 1:
        raise ValueError('GKD gkd_logits_topk must be >= 1 when set.')


def _check_auxiliary_adapters(cfg: 'RLHFConfig') -> None:
    if len(cfg.ref_adapters) > 1:
        raise ValueError('ref_adapters currently supports one frozen reference adapter.')
    if len(cfg.teacher_adapters) > 1:
        raise ValueError('teacher_adapters currently supports one frozen teacher adapter.')
    if cfg.teacher_adapters and cfg.teacher_model is None:
        raise ValueError('teacher_adapters requires a distinct local teacher_model.')
    reward_models = cfg.reward_model or []
    reward_fields = {
        'reward_adapters': cfg.reward_adapters,
        'reward_model_type': cfg.reward_model_type,
        'reward_model_revision': cfg.reward_model_revision,
        'reward_template': cfg.reward_template,
    }
    for field, values in reward_fields.items():
        if values and not reward_models:
            raise ValueError(f'{field} requires reward_model.')
        if values and len(values) != len(reward_models):
            raise ValueError(f'{field} must contain exactly one value per reward_model.')
    if cfg.rlhf_type not in ('grpo', 'rft') and cfg.reward_template:
        raise ValueError('reward_template is supported by GRPO and RFT only.')


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

    ``use_galore`` is transformers-only (it is in _HF_ONLY, so _check_backend_specific already rejects
    it on Megatron). GaLore projects the FULL-PARAMETER gradient into a low-rank subspace, which is a
    different low-rank approximation than an adapter, so three combinations are fatal here rather than
    silently degraded:
      - GaLore alongside an adapter (a non-None TunerConfig -- ``select_tuner`` maps ``tuner='full'`` to
        None, so any config here means an adapter is applied): stacking two low-rank maps trains their
        product, not either method. GaLore is a full-parameter technique.
      - the legacy GaLore knobs twinkle's GaLoreConfig does not implement (per-parameter optimizers,
        the with_embedding switch): accepting them would train without them. QGaLore (quantized
        projection, ``galore_quantization`` + its knobs) IS implemented -- twinkle resolves it to
        QGaLoreAdamW8bit -- so it is not in this list.
      - a base optim with no GaLore variant (Adam / SGD / muon), or a quantized projection on a base
        with no QGaLore variant (anything but AdamW): resolve_galore_target refuses both, so the error
        surfaces here, before the weights load, rather than deep in configure_optimizer.
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
    from swift.dev.naming import resolve_galore_target, resolve_optim_target
    try:
        resolve_galore_target(resolve_optim_target(train_config.optim)[0], quantize=train_config.galore_quantization)
    except NotImplementedError as exc:
        raise ValueError(str(exc)) from exc


def _check_muon(train_config: 'TrainConfig', distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Reject muon pairings that Megatron itself refuses, or that would train something else.

    Mirrors legacy megatron_args.py::_check_muon with one deliberate difference: legacy silently sets
    ``use_distributed_optimizer = False`` when muon is selected, and this raises instead. The knob is
    one the user typed, and turning off the distributed optimizer changes both the memory profile and
    what a checkpoint contains -- the same class of hidden downgrade as the padding_free case above.

    The mcore version gate is legacy's too. Checking it here means a CLI mistake fails on the driver;
    it is safe to read on this side because it is a package version rather than a device property, so
    unlike the FP8/Blackwell checks it does not describe hardware this process may not have.

    The transformers backend selects muon differently -- with ``optim='muon'`` (twinkle MuonClip), not
    the Megatron-only ``optimizer`` knob -- so its guards live in the ``not is_megatron`` branch below.
    """
    if not is_megatron:
        # Point a Megatron `optimizer='muon'` mistake at the transformers spelling instead of letting
        # it be silently ignored (the Megatron branch is the only one that reads `optimizer`).
        if 'muon' in train_config.optimizer:
            raise ValueError(
                f'TrainConfig.optimizer={train_config.optimizer!r} is a Megatron optimizer, but the active '
                'backend is transformers. Use TrainConfig.optim for the transformers path, or switch '
                'DistributedConfig.backend.')
        # muon_momentum / muon_use_nesterov / muon_num_ns_steps are dual-backend (MegatronOptimizer on
        # Megatron, MuonClip's MuonConfig here), but on transformers they are read ONLY when
        # optim='muon'. The Megatron-specific muon_* fields stay in _MEGATRON_ONLY, so
        # _check_backend_specific already refuses those here; this guards the three dual ones against
        # being silently ignored by a non-muon optim.
        if train_config.optim.lower() != 'muon':
            stray = [n for n in ('muon_momentum', 'muon_use_nesterov', 'muon_num_ns_steps')
                     if n in _changed_fields(train_config)]
            if stray:
                raise ValueError(
                    f'{stray} configure the muon optimizer, but TrainConfig.optim={train_config.optim!r} '
                    "on the transformers backend, so they would be ignored. Set optim='muon' to use "
                    'them, or drop them.')
        return

    if 'muon' not in train_config.optimizer:
        return

    from swift.dev.naming import mcore_version_at_least
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


def _check_unsloth_strategy(tuner_config: Optional['TunerConfig'], distributed_config: 'DistributedConfig',
                            template_config: 'TemplateConfig') -> None:
    """Refuse the strategies unsloth cannot co-exist with.

    unsloth rebuilds the module graph around a causal-LM checkpoint and compiles its own Triton
    kernels/RoPE cache, so a strategy that shards the parameters (DeepSpeed ZeRO / FSDP) or splits the
    sequence across ranks (Ulysses SP) either wraps a graph unsloth has already rewritten or feeds its
    fused kernels a sequence shard they do not expect. Each combination is refused here rather than left
    to crash inside unsloth's patcher or silently train an unsharded replica.
    """
    if tuner_config is None or getattr(tuner_config, 'tuner_backend', None) != 'unsloth':
        return
    if distributed_config.deepspeed:
        raise NotImplementedError(
            'tuner_backend="unsloth" cannot run under DeepSpeed: unsloth installs its own kernels and module '
            'graph and does not compose with a ZeRO strategy. Drop --deepspeed, or use tuner_backend="peft".')
    if distributed_config.fsdp:
        raise NotImplementedError(
            'tuner_backend="unsloth" cannot run under FSDP: unsloth installs its own kernels and module graph '
            'and does not compose with parameter sharding. Drop --fsdp, or use tuner_backend="peft".')
    if template_config.sequence_parallel_size > 1:
        raise NotImplementedError(
            f'tuner_backend="unsloth" cannot run under sequence_parallel_size='
            f'{template_config.sequence_parallel_size}: unsloth\'s fused kernels assume a whole sequence per '
            'rank. Set sequence_parallel_size=1, or use tuner_backend="peft".')


def _check_deepspeed_autotp(distributed_config: 'DistributedConfig') -> None:
    """DeepSpeed AutoTP is not implemented on the twinkle strategy, so refuse the knob rather than drop it.

    AutoTP needs two things the twinkle DeepSpeed path does not provide: tensor-parallel groups built from
    ``tensor_parallel.autotp_size`` in the config, AND a data sampler that treats each TP group as one data
    rank (legacy shards its BatchSamplerShard with ``tp_size=autotp_size``). Injecting only the config key
    would leave every rank in a TP group fed different data -- a silent correctness bug -- so the knob is
    rejected on both backends instead of half-wired. (It is deliberately NOT in _HF_ONLY: that table's
    "only implemented by transformers" message would be wrong, since neither backend implements it.)
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
    """Guards for the in-training generative eval path (``predict_with_generate`` -> EvalScope).

    ``predict_with_generate`` is transformers-only (it is in _HF_ONLY, so _check_backend_specific already
    refused it on Megatron before this runs). Four combinations cannot evaluate the way they claim and are
    refused here rather than producing a number from a partial/wrong setup:
      - no ``eval_dataset``: the generative path runs an EvalScope benchmark, so there is nothing to serve
        without one (the validation-loss path uses the split-off validation set instead).
      - transformers sampler under sequence parallelism: HF ``.generate()`` runs the whole sequence on one
        rank and is unaware of the ulysses/ring shard, so it would generate from a partial sequence.
      - vllm/sglang sampler outside ray: those backends weight-sync into a co-resident engine through a Ray
        DeviceGroup (CheckpointEngineManager), which a local/torchrun launch has no way to place.
      - transformers sampler under a parameter-sharding strategy in a multi-rank local/torchrun launch: only
        rank 0 drives EvalScope while its peers park at the eval barrier, so a sharded forward's all-gather
        never completes and the run hangs (see the guard below).
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
    # rank0-only generation is safe only while the model's forward is rank-local. A parameter-sharding
    # strategy rebuilds each weight through a collective all-gather inside forward; under torchrun the
    # peers park at the eval barrier (dist.barrier in _evaluate_generate), so rank 0's all-gather never
    # completes and the run hangs at the first generative eval. Only ZeRO-3 shards params -- ZeRO-1/2 and
    # DDP replicate them and stay rank-local -- so those are allowed (the loop warns about their 1/N
    # throughput at runtime). Refuse here, before ranks spawn, rather than deadlock mid-training.
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

    Duplicated on purpose with MegatronStrategy._check_fsdp, which is the authority: that one runs in
    the process that builds the model, so it also covers cookbook users who never touch dev's config
    layer. Checking here as well means a CLI typo fails on the driver, before ranks are spawned.
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


#: (format field, param-gather field) for each low-precision format dev exposes. The amax knobs are
#: deliberately absent: their defaults are non-None, so "did the user set this?" is unanswerable and
#: a dependency check on them would fire on every run.
_QUANT_FORMATS = (('fp4_format', 'fp4_param_gather'), ('fp8_format', 'fp8_param_gather'))


def _check_quantization(model_config: 'ModelConfig', distributed_config: 'DistributedConfig',
                        is_megatron: bool) -> None:
    """Reject FP4/FP8 settings that cannot do what they say.

    Errors rather than warnings because every case below starts, reports a normal-looking loss, and
    trains nothing or trains something other than what was asked for.

    The environment preconditions (Blackwell for NVFP4, a TE new enough for the chosen recipe) are
    deliberately NOT checked here: this runs on the driver, which in Ray mode is not the process --
    nor necessarily the node -- that builds the model, so a check here would test the wrong GPU.
    mcore-bridge's ModelConfig checks them where the model is actually built.
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
            # DistributedOptimizer._copy_main_params_to_model_params is the only code that
            # re-quantizes the FP32 master shards back into the quantized parameters. Under any other
            # optimizer they keep their initial values for the whole run while the loss is computed
            # from them, so it neither errors nor learns. megatron asserts the same thing on its own
            # CLI ('--fp8-param-gather only supported with distributed optimizer, ...').
            raise ValueError(
                f'ModelConfig.{param_gather} requires DistributedConfig.use_distributed_optimizer=True: '
                'quantized parameters are updated by re-quantizing the distributed optimizer\'s master shards, '
                'and no other optimizer implements that step, so the model would never change.')

    if len(active) > 1:
        # megatron enters exactly one quantization context per transformer layer and its own
        # TransformerConfig raises on this; caught here so it fails on the driver, before a model is
        # built on every rank.
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

    if getattr(tuner_config, 'tuner_backend', None) == 'unsloth' and method != 'bnb':
        raise ValueError(
            f'Unsloth only exposes load_in_4bit/load_in_8bit for BNB, so quant_method={method!r} is unsupported. '
            'Use --quant_method bnb or the default tuner backend.')

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

    Every failure here is one that would otherwise be silent -- an MTP run that trains nothing, or
    exports no MTP layer -- which is why these are errors rather than warnings. The only exception is
    LoRA, which *can* work if the adapter covers the MTP modules, so it warns instead.
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
        # Not an error: this is trainable if the adapter targets the MTP modules, which we cannot
        # decide from target_modules alone (it may be 'all-linear', or name them explicitly).
        # twinkle re-checks against the built model and warns if nothing ended up trainable.
        logger.warning('enable_mtp_training with tuner=%r: the MTP layers are base parameters, so they only '
                       'train if the adapter covers them. Otherwise the MTP loss is computed and discarded.',
                       tuner_config.tuner)


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
    from swift.dev.optimizer import megatron_weight_decay_bounds, warmup_budget

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
    # ordinary cosine -- the name silently not doing what it says. (On the HF path transformers
    # itself raises when neither min_lr nor min_lr_rate is given.)
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
    ('model_config', 'vit_attn_impl', None),
    ('model_config', 'language_model_only', False),
    ('dataset_config', 'data_sharding', False),
    # NOTE: clip_grad is NOT here -- CLI normalization folds it into max_grad_norm for both backends.
    ('train_config', 'weight_decay_incr_style', 'constant'),
    ('train_config', 'start_weight_decay', None),
    ('train_config', 'end_weight_decay', None),
    ('train_config', 'min_lr', 0.0),
    ('train_config', 'optimizer', 'adam'),
    ('train_config', 'sgd_momentum', 0.9),
    # muon_momentum / muon_use_nesterov / muon_num_ns_steps are deliberately NOT here: they are
    # dual-backend (MegatronOptimizer reads them here, MuonClip's MuonConfig reads them on the
    # transformers path), so _check_backend_specific must not reject them on transformers. The
    # remaining muon_* fields are Megatron-implementation-specific (split_qkv / scale_mode / tp_mode
    # / ... have no MuonConfig counterpart) and stay transformers-rejected. _check_muon additionally
    # refuses the three dual ones on transformers unless optim='muon' actually consumes them.
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
    ('train_config', 'finetune', True),
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

_HF_ONLY = (
    ('model_config', 'experts_impl', None),
    ('model_config', 'new_special_tokens', []),
    ('model_config', 'device_map', None),
    ('model_config', 'max_memory', None),
    ('model_config', 'local_repo_path', None),
    ('model_config', 'model_kwargs', None),
    ('model_config', 'init_strategy', None),
    ('distributed_config', 'deepspeed', None),
    ('distributed_config', 'zero_hpz_partition_size', None),
    ('distributed_config', 'fsdp', None),
    ('distributed_config', 'ddp_find_unused_parameters', None),
    ('train_config', 'use_galore', False),
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

    Mirrors legacy megatron_args.py:799-800. Selective recomputation always targets the same
    (attention) operations, so it has no layer partitioning to configure -- `recompute_method`
    (uniform/block) only means something under 'full'. Pairing them asks Megatron to partition a mode
    that is not partitioned, which it refuses; catching it here reports the contradiction before ranks
    are spawned rather than deep in the model build.
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
    """The per-stage decoder layer overrides need a pipeline to distribute across.

    Mirrors legacy megatron_args.py:832-835. `decoder_first_pipeline_num_layers` /
    `decoder_last_pipeline_num_layers` move layers onto the first/last pipeline stage to balance an
    uneven split; with `pipeline_model_parallel_size == 1` there is only one stage, so both are the
    whole model and the override describes a partition that does not exist.
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

    The ratio freezes the leading fraction of parameters by cumulative element count in
    ``named_parameters()`` order. Under PP>1 each pipeline rank holds a DIFFERENT set of layers, so
    "the first N% by element count" selects a different, meaningless slice on each stage -- and the
    freeze runs per rank (twinkle's remote seam has no cross-stage view to make it consistent). Mirrors
    legacy Megatron-SWIFT, which documents the two as mutually exclusive. Name-prefix / regex freeze is
    unaffected (it matches by name on whatever a rank holds) and stays allowed.

    Only the full-parameter path consumes the ratio (tuner_config is None); an adapter run ignores it,
    so there is nothing to reject there.
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
    """Tensor-parallel comm/GEMM overlap only exists when sequence parallelism splits the activations.

    Mirrors legacy megatron_args.py:896-898. The overlap hides the tensor-parallel all-gather/
    reduce-scatter behind the GEMM, and those collectives only appear when `sequence_parallel` shards
    the activations along the sequence; without it there is nothing to overlap and Megatron asserts the
    same pairing.
    """
    if not is_megatron:
        return
    if distributed_config.tp_comm_overlap and not distributed_config.sequence_parallel:
        raise ValueError('DistributedConfig.tp_comm_overlap requires sequence_parallel=True: the overlap hides '
                         'the tensor-parallel collectives that only exist under sequence parallelism, so with it '
                         'off there is nothing to overlap.')


def _check_sequence_parallel_tp(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Sequence parallelism splits activations across the tensor-parallel ranks, so it needs TP > 1.

    legacy silently sets `sequence_parallel = False` here (megatron_args.py:890-891). dev raises
    instead, for the reason the muon and attn-backend guards give: `sequence_parallel` is one the user
    typed, and quietly turning it off changes the activation memory profile so a run can look fine and
    use far more memory than the config implies. With `tensor_model_parallel_size == 1` there are no
    tensor-parallel ranks to split across, so the flag cannot do anything.
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
                              rlhf_config: Optional['RLHFConfig'],
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
    if rlhf_config is not None and rlhf_config.rlhf_type in {'grpo', 'gkd', 'ppo', 'opsd', 'mopd', 'rft'}:
        if 'ignore_data_skip' in changed:
            raise NotImplementedError(
                f'ignore_data_skip does not apply to {rlhf_config.rlhf_type} because its online loop has no resumable '
                'dataset iterator. Remove the option.')
        if rlhf_config.rlhf_type == 'ppo' and checkpoint_config.save_total_limit == 1:
            raise ValueError(
                'PPO requires save_total_limit >= 2 because the policy and value-model components are saved '
                'sequentially; retaining one previous complete checkpoint avoids a no-valid-checkpoint window.')


def _check_save_total_limit(checkpoint_config: Optional['CheckpointConfig'], is_megatron: bool) -> None:
    """Validate the rolling checkpoint limit and preserve legacy Megatron's stricter lower bound.

    `async_save` writes in the background, and the limit's delete-oldest step cannot tell whether an
    in-flight async save has finished, so the two are incompatible.
    """
    if checkpoint_config is None or checkpoint_config.save_total_limit is None:
        return
    if checkpoint_config.save_total_limit < 1:
        raise ValueError('CheckpointConfig.save_total_limit must be >= 1.')
    if not is_megatron:
        return
    if checkpoint_config.async_save:
        raise ValueError('CheckpointConfig.save_total_limit is incompatible with async_save=True: the rolling '
                         'delete of old checkpoints cannot tell whether a background save has finished. Disable '
                         'one of the two.')
    if checkpoint_config.save_total_limit < 2:
        raise ValueError(
            'CheckpointConfig.save_total_limit must be >= 2 on the Megatron backend, matching the legacy CLI.')


#: rlhf_type -> whether it trains against a separate reference model. CPO/ORPO fold the reference into
#: their own loss and LoRA uses the adapter-disabled base as reference, so neither takes a ref_model.
_RLHF_USES_REF_MODEL = ('dpo', 'kto', 'ppo', 'grpo')


def _check_rlhf_ref_model(model_config: 'ModelConfig', tuner_config: Optional['TunerConfig'],
                          rlhf_config: Optional['RLHFConfig']) -> None:
    """Reject a reference model passed to an algorithm that has none.

    Mirrors the trailing `elif self.ref_model is not None: raise` of legacy rlhf_args.py:297-298. The
    derivation half (defaulting ref_model to model for the algorithms that use one) lives in
    process.py::_derive_rlhf_ref_model; this is the refusal half. CPO/ORPO build the reference into
    their loss and LoRA training uses the base model with the adapter disabled, so a `--ref_model`
    there is a knob that would be silently ignored -- the class of mistake validate.py exists to catch.
    """
    if rlhf_config is None:
        return
    rlhf_type = getattr(rlhf_config, 'rlhf_type', None)
    tuner = getattr(tuner_config, 'tuner', 'full') if tuner_config is not None else 'full'
    uses_ref = rlhf_type in _RLHF_USES_REF_MODEL and (tuner == 'full' or bool(rlhf_config.ref_adapters))
    if rlhf_config.ref_model is None:
        if rlhf_config.ref_adapters:
            raise ValueError('ref_adapters requires a reference model; call process_configs or set ref_model.')
        return
    # grpo with beta=0 drops the KL term, so even a ref-using algorithm needs no reference then.
    if rlhf_type == 'grpo' and rlhf_config.beta == 0.0:
        uses_ref = False
    if not uses_ref:
        raise ValueError(f'RLHFConfig.ref_model={rlhf_config.ref_model!r} is not used by rlhf_type={rlhf_type!r}'
                         f' with tuner={tuner!r}: CPO/ORPO fold the reference into their loss and LoRA '
                         'uses the adapter-disabled base as the reference, so no separate ref_model is loaded. '
                         'Remove it.')


#: rlhf_types whose training loss path is verified correct under the packed / variable-length layout that
#: padding_free (and packing, which implies it) produces. The transformers forward normalizes every packed
#: micro batch back to per-sequence ``[num_seq, max_seq]`` via ``processor.unpack_packed_sequences`` BEFORE
#: calculate_loss, so a loss reading per-token logps/labels off that unpacked frame is packing-agnostic:
#:   - grpo / rft: online RL -- GRPOLoss, and rft's plain cross_entropy (the SFT loss), over the unpacked
#:     frame; both share GRPOLoop's rollout + micro-step path.
#:   - dpo / kto / cpo / orpo / simpo: the offline PreferenceLossBase family -- all five subclass it and
#:     consume ``_split_chosen_rejected`` + ``_compute_sequence_logps``/``_compute_avg_logps`` over the same
#:     unpacked per-token logps, on the one run_dpo encode path (dpo/kto were the legacy-verified seeds).
#:   - gkd: teacher-student divergence over the unpacked labels/logits.
_RL_PADDING_FREE_WIRED_TYPES = ('grpo', 'rft', 'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'gkd')

#: rlhf_types refused BY NAME (never silently) because one component of their path is not packed-safe.
_RL_PADDING_FREE_REFUSED_REASONS = {
    'ppo':
    "PPO's per-token value-head critic does not cover the packed / variable-length layout (run_ppo.py "
    'module docstring), so the critic would read a flattened row as one sequence.',
    'rm':
    'RewardLoss reads the seq_cls head\'s pooled [B, 1] score, and pooling a flattened packed row does not '
    'split back into one score per packed sub-sequence.',
    'opsd':
    "OPSD's teacher->student per-token logps alignment across two independently-unpacked forwards is not "
    'verified under variable-length packing.',
    'mopd':
    "MOPD fuses K teacher logps channels onto the student frame; that K-way alignment across two "
    'independently-unpacked forwards is not verified under variable-length packing.',
}


def _check_rlhf_padding_free(template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
                            rlhf_config: Optional['RLHFConfig']) -> None:
    """Only some RLHF algorithms have a padding-free/packing training path.

    padding_free (and packing, which implies it) flattens a micro batch into one variable-length sequence;
    the forward unpacks it back to per-sequence rows before calculate_loss, so any loss over that unpacked
    per-token frame is correct (``_RL_PADDING_FREE_WIRED_TYPES``). The refused types are named explicitly
    with the component that is not packed-safe rather than silently mis-computed later. Widened from
    legacy rlhf_args.py::_check_padding_free (grpo/dpo/kto/gkd) after verifying cpo/orpo/simpo share DPO's
    PreferenceLossBase and rft shares GRPO's rollout with the SFT cross_entropy loss.
    """
    if rlhf_config is None:
        return
    if not (template_config.padding_free or dataset_config.packing):
        return
    rlhf_type = getattr(rlhf_config, 'rlhf_type', None)
    if rlhf_type in _RL_PADDING_FREE_WIRED_TYPES:
        return
    feature = 'packing' if dataset_config.packing else 'padding_free'
    reason = _RL_PADDING_FREE_REFUSED_REASONS.get(rlhf_type)
    if reason is not None:
        raise ValueError(f'rlhf_type={rlhf_type!r} does not support {feature}: {reason} '
                         'Set the corresponding flag to False.')
    raise ValueError(f'rlhf_type={rlhf_type!r} does not support {feature}: only '
                     f'{"/".join(_RL_PADDING_FREE_WIRED_TYPES)} implement the variable-length training path it '
                     'produces. Set the corresponding flag to False.')


#: attn_impl values whose attention kernel handles the variable-length (THD) layout that
#: padding_free produces under Ulysses SP. Mirrors legacy sft_args.py supported_impls; twinkle's
#: SP strategy enforces the same requirement at first forward (flash_attention_2/3 only).
_SP_PADDING_FREE_ATTN_IMPLS = ('flash_attn', 'flash_attention_2', 'flash_attention_3', 'flash_attention_4')

#: Online-RL rlhf_types whose Ulysses SP consumer wiring is in place: run_grpo/run_rft call
#: ``assembly.plan_sp_mesh()`` before build_model (installing the dp x ulysses ray mesh the trainable
#: policy, its colocated sampler and the ``data_world_size`` batch-width formula all share). PPO is
#: deliberately absent -- its per-token value-head critic runs WITHOUT SP/packed layouts (run_ppo.py
#: module docstring), so a SP policy paired with an SP=1 critic would slice_dp each forward_backward at a
#: different data_world_size and starve the critic's ranks. Offline preference types (dpo/kto/...) are
#: absent too: they are not an "under ray" recipe, so W38 does not wire them.
_RL_SP_WIRED_TYPES = ('grpo', 'rft')


def _check_sp_hf_feasibility(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                            dataset_config: 'DatasetConfig', distributed_config: 'DistributedConfig',
                            is_megatron: bool, sp: int) -> None:
    """Backend/layout feasibility guards shared by the SFT (local) and online-RL (ray) Ulysses SP paths.

    These are facts about the transformers backend and the batch layout, independent of HOW the ranks are
    launched, so both SP callers reuse them and only the world-size source differs (SFT reads
    ``Platform.get_world_size()``; online RL reads ``DistributedConfig.nproc_per_node``). Every branch
    raises: SP that cannot do what the config says would otherwise SILENTLY train with SP=1 or crash deep
    in the first forward.
    """
    if is_megatron:
        # TemplateConfig.sequence_parallel_size is the HF Ulysses knob; Megatron's sequence
        # parallelism is DistributedConfig.sequence_parallel (TP-SP), a different feature.
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
                                 rlhf_config: Optional['RLHFConfig'], is_megatron: bool) -> None:
    """Ulysses SP (TemplateConfig.sequence_parallel_size) for the online-RL recipes under ray.

    Only ``grpo``/``rft`` are wired (``_RL_SP_WIRED_TYPES``): their recipes plan the dp x ulysses ray mesh
    via ``assembly.plan_sp_mesh()`` and read ``data_world_size`` off it for the batch width, so the config
    and the run agree. Every other rlhf_type raises rather than silently training with SP=1 -- PPO because
    its critic cannot follow the policy onto the SP mesh, the offline preference family because W38 wires
    only the "under ray" path. Feasibility reuses the SFT guards; the world here is ``nproc_per_node`` (the
    ray 'model' group size), NOT ``Platform.get_world_size()`` -- the driver orchestrates and has no
    torchrun WORLD_SIZE.
    """
    if rlhf_config is None or template_config.sequence_parallel_size <= 1:
        return
    sp = template_config.sequence_parallel_size
    rlhf_type = getattr(rlhf_config, 'rlhf_type', None)
    if rlhf_type == 'ppo':
        raise ValueError(f'rlhf_type="ppo" does not support sequence_parallel_size={sp}: PPO\'s per-token '
                         'value-head critic runs without sequence-parallel / packed layouts, so a SP policy '
                         'and an SP=1 critic would slice_dp each forward_backward at a different '
                         'data_world_size and starve the critic\'s ranks. Set sequence_parallel_size=1 for PPO.')
    if rlhf_type not in _RL_SP_WIRED_TYPES:
        raise ValueError(f'rlhf_type={rlhf_type!r} does not support sequence_parallel_size={sp} on the dev path: '
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
                                is_megatron: bool, rlhf_config: Optional['RLHFConfig'] = None) -> None:
    """Guards for Ulysses sequence parallelism (TemplateConfig.sequence_parallel_size) on the HF SFT path.

    Every check raises: SP that cannot do what the config says would otherwise SILENTLY train with
    SP=1 (nothing on the HF path used to read this knob) or crash deep in the first forward.
    ``process_configs`` has already resolved packing-derived padding_free, so the flash-attn guard sees the
    effective value. Streaming is allowed: twinkle's IterableFetcher slices by data_world_size the same way.

    RLHF runs are delegated to :func:`_check_rlhf_sequence_parallel` (the online grpo/rft ray path uses
    ``nproc_per_node`` as its world, and PPO / offline types are rejected there), so this returns early for
    them -- otherwise the ``mode != 'local'`` guard below would double-reject the ray RL path.
    """
    sp = template_config.sequence_parallel_size
    if sp <= 1:
        return
    if rlhf_config is not None:
        return

    _check_sp_hf_feasibility(model_config, template_config, dataset_config, distributed_config, is_megatron, sp)

    if distributed_config.mode != 'local':
        raise NotImplementedError(f'sequence_parallel_size={sp} is only wired for mode="local" (torchrun): under '
                                  "mode='ray' the model gets a pure data-parallel mesh (_apply_ray_placement), so "
                                  'SP would silently not apply. Run with torchrun, or set sequence_parallel_size=1.')

    # twinkle.initialize(mode='local') never calls dist.init_process_group -- world size comes from
    # Platform.get_world_size() (the WORLD_SIZE env torchrun sets), the same source initialize uses
    # to build the default mesh. Requiring dist here would reject every real SP run.
    from twinkle.utils import Platform
    world = Platform.get_world_size()
    if world < 2:
        raise ValueError(f'sequence_parallel_size={sp} requires torchrun (WORLD_SIZE>=2): a single process has no '
                         'ranks to split a sequence across. Set sequence_parallel_size=1 for a single-process run.')
    if world % sp != 0:
        raise ValueError(f'world_size={world} is not divisible by sequence_parallel_size={sp}: the SP groups must '
                         'tile the ranks exactly. Adjust the rank count or sequence_parallel_size.')

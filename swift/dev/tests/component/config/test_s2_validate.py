# Copyright (c) ModelScope Contributors. All rights reserved.
"""S2b -- every validation refusal in config/validate.py (ut_plan S2-27..S2-56).

One behaviour per test, each refusing exactly the combination its docstring names, with the
message matched so a rewrite cannot lose the guidance. The three moved files close their slices:
test_hf_sp_config.py (S2-44), test_quantization.py (S2-48 half), test_tracking.py (S2-60);
_check_mtp's 16 cases arrive with the megatron suite (S2-49 / S13). Test ids in docstrings.
"""
from __future__ import annotations

import pytest

from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    RewardConfig,
    RLConfig,
    RolloutConfig,
    TemplateConfig,
    TrainConfig,
    TunerConfig,
)
from swift.dev.config.validate import (
    _check_async_mode,
    _check_auxiliary_adapters,
    _check_checkpoint_runtime,
    _check_channel_companions,
    _check_chord,
    _check_deepspeed_autotp,
    _check_distillation,
    _check_dpo_loss,
    _check_dynamic_sampling,
    _check_eval_generation,
    _check_eval_iters,
    _check_galore,
    _check_grpo_cached_dataset,
    _check_grpo_controls,
    _check_grpo_truncation_strategy,
    _check_group_by_length,
    _check_lazy_tokenize,
    _check_logging,
    _check_megatron_attn_backend,
    _check_megatron_fsdp,
    _check_megatron_recompute,
    _check_muon,
    _check_param_groups,
    _check_pipeline_decoder_layers,
    _check_preference_reference,
    _check_prm,
    _check_quantization,
    _check_reference_sync,
    _check_rlhf_advanced,
    _check_rlhf_padding_free,
    _check_rlhf_padding_side,
    _check_rlhf_ref_model,
    _check_router_replay,
    _check_sampling_replay,
    _check_save_total_limit,
    _check_selective_recompute,
    _check_sequence_parallel_tp,
    _check_streaming_publication,
    _check_tp_comm_overlap,
    _check_unsloth_strategy,
    _sync_only_feature,
    _streams_under_none,
    validate_infer_config,
    validate_multi_turn_config,
    validate_rollout_config,
)


def _explicit(config, *names):
    config._explicit_fields = set(getattr(config, '_explicit_fields', set())) | set(names)
    return config


# --- RL advanced knobs (S2-27..S2-37) ----------------------------------------------------


def test_rlhf_advanced_grpo_only_gates():
    """S2-27: chord/advantage_reweight/sdar/dynamic_sample are GRPO-only; sync_ref_model needs
    grpo or ppo (the only reference-bearing algorithms)."""
    with pytest.raises(ValueError, match='GRPO-only'):
        _check_rlhf_advanced(TrainConfig(), RLConfig(rl_type='dpo', chord_sft_dataset='d'))
    with pytest.raises(ValueError, match='GRPO-only'):
        _check_rlhf_advanced(TrainConfig(), RLConfig(rl_type='dpo', advantage_reweight='rlsd'))
    with pytest.raises(ValueError, match='GRPO-only'):
        _check_rlhf_advanced(TrainConfig(), RLConfig(rl_type='kto', sdar_loss_coef=0.1))
    with pytest.raises(ValueError, match='GRPO-only'):
        _check_rlhf_advanced(TrainConfig(), RLConfig(rl_type='dpo', dynamic_sample=True))
    with pytest.raises(ValueError, match='grpo or ppo'):
        _check_rlhf_advanced(TrainConfig(), RLConfig(rl_type='dpo', sync_ref_model=True))
    # a GRPO run with none of the knobs passes the gate (the later reward-channel guard needs a
    # channel; the deeper checks run on their own below)
    _check_rlhf_advanced(TrainConfig(), RLConfig(rl_type='grpo', num_generations=4),
                         RewardConfig(orm=['accuracy']))


def test_dpo_loss_validation():
    """S2-28: DPO's f_divergence must be one of the declared types; loss_weights must give one
    weight per loss_type; non-dpo runs carry the knobs through untouched."""
    from swift.dev.config.validate import _DPO_F_DIVERGENCE_TYPES
    good = _DPO_F_DIVERGENCE_TYPES[0] if isinstance(_DPO_F_DIVERGENCE_TYPES, (list, tuple)) \
        else sorted(_DPO_F_DIVERGENCE_TYPES)[0]
    _check_dpo_loss(RLConfig(rl_type='dpo', f_divergence_type=good))  # a declared type passes

    cfg = RLConfig(rl_type='dpo', f_divergence_type='not_a_divergence')
    with pytest.raises(ValueError, match='f_divergence_type'):
        _check_dpo_loss(cfg)

    cfg = RLConfig(rl_type='dpo', loss_type=['sigmoid', 'hinge'], loss_weights=[1.0])
    with pytest.raises(ValueError, match='one weight per'):
        _check_dpo_loss(cfg)

    cfg = RLConfig(rl_type='dpo', loss_type=['sigmoid', 'hinge'], loss_weights=[0.5, 0.5])
    _check_dpo_loss(cfg)

    cfg = RLConfig(rl_type='grpo', f_divergence_type='bogus')  # dpo-only knob on another run: skipped
    _check_dpo_loss(cfg)


def test_kto_reference_free_refused():
    """S2-29: KTO's reward AND z_KL are both log-ratios vs the reference, so reference-free KTO
    cannot be honoured; every other reference-free algorithm passes."""
    with pytest.raises(ValueError, match='KTO cannot be reference-free'):
        _check_preference_reference(RLConfig(rl_type='kto', reference_free=True))
    _check_preference_reference(RLConfig(rl_type='dpo', reference_free=True))


@pytest.mark.parametrize('kwargs, message', [
    (dict(rl_type='dpo', log_completions=True), 'GRPO-only'),
    (dict(rl_type='dpo', num_iterations=2), 'GRPO-only'),
    (dict(rl_type='dpo', loss_type=['fipo']), 'GRPO-only'),  # a GRPO variant on a non-GRPO run
    (dict(rl_type='grpo', num_generations=1), 'num_generations >= 2'),
    (dict(rl_type='grpo', num_iterations=0), 'must be > 0'),
    (dict(rl_type='grpo', top_entropy_quantile=1.5), r'\(0, 1\]'),
    (dict(rl_type='grpo', loss_type=['bogus']), 'Unsupported GRPO loss_type'),
    (dict(rl_type='grpo', loss_type=['grpo', 'dapo']), 'exactly one loss_type'),
    (dict(rl_type='grpo', loss_type=['dro'], dro_beta=0.0), 'dro_beta > 0'),
    (dict(rl_type='grpo', loss_type=['clip_cov'], clip_cov_lb=1.0, clip_cov_ub=0.5), 'clip_cov_lb < clip_cov_ub'),
])
def test_grpo_controls_matrix(kwargs, message):
    """S2-30: the GRPO control/loss-type/knobs matrix -- GRPO-only knobs on other runs, the
    group-relative baseline bound, positive-control bounds, and the per-variant knob pairs."""
    kwargs.setdefault('num_generations', 4)
    with pytest.raises(ValueError, match=message):
        _check_grpo_controls(RLConfig(**kwargs))


def test_grpo_loss_type_gspo_is_sequence_level():
    """S2-30 (GSPO): gspo IS sequence-level IS; an explicit token-level request is refused while
    the promoted default passes."""
    cfg = _explicit(RLConfig(rl_type='grpo', num_generations=4, loss_type=['gspo'],
                             importance_sampling_level='token'), 'importance_sampling_level')
    with pytest.raises(ValueError, match='contradicts'):
        _check_grpo_controls(cfg)
    cfg = RLConfig(rl_type='grpo', num_generations=4, loss_type=['gspo'],
                   importance_sampling_level='sequence')
    _check_grpo_controls(cfg)


def test_grpo_cached_dataset_and_truncation_refusals():
    """S2-31: GRPO rolls out from raw messages (cached_dataset would be ignored) and truncates
    prompts from the left only ('right' cuts what the completion conditions on)."""
    with pytest.raises(ValueError, match='cached_dataset is not supported for GRPO'):
        _check_grpo_cached_dataset(DatasetConfig(cached_dataset=['c']), RLConfig(rl_type='grpo'))
    _check_grpo_cached_dataset(DatasetConfig(cached_dataset=['c']), RLConfig(rl_type='dpo'))

    with pytest.raises(ValueError, match="truncation_strategy 'left' or 'delete'"):
        _check_grpo_truncation_strategy(TemplateConfig(truncation_strategy='right'), RLConfig(rl_type='grpo'))
    with pytest.raises(ValueError, match="truncation_strategy 'left' or 'delete'"):
        _check_grpo_truncation_strategy(TemplateConfig(truncation_strategy='split'), RLConfig(rl_type='grpo'))
    _check_grpo_truncation_strategy(TemplateConfig(truncation_strategy='left'), RLConfig(rl_type='grpo'))
    _check_grpo_truncation_strategy(TemplateConfig(truncation_strategy='right'), RLConfig(rl_type='dpo'))


def test_dynamic_sampling_requires_orm_and_variance():
    """S2-32: dynamic_sample filters by reward variance, so it needs the orm channel and >= 2
    generations per prompt, and the resample budget must be positive."""
    with pytest.raises(ValueError, match='orm channel'):
        _check_dynamic_sampling(RLConfig(rl_type='grpo', dynamic_sample=True, num_generations=4))
    with pytest.raises(ValueError, match='orm channel'):
        _check_dynamic_sampling(RLConfig(rl_type='grpo', dynamic_sample=True, num_generations=4), RewardConfig())
    with pytest.raises(ValueError, match='num_generations >= 2'):
        _check_dynamic_sampling(RLConfig(rl_type='grpo', dynamic_sample=True, num_generations=1),
                                RewardConfig(orm=['accuracy']))
    with pytest.raises(ValueError, match='max_resample_times'):
        _check_dynamic_sampling(RLConfig(rl_type='grpo', dynamic_sample=True, num_generations=4,
                                         max_resample_times=0), RewardConfig(orm=['accuracy']))
    _check_dynamic_sampling(RLConfig(rl_type='grpo', dynamic_sample=True, num_generations=4),
                            RewardConfig(orm=['accuracy']))


def test_reference_sync_bounds_and_parallel_spec():
    """S2-33: EMA sync needs beta>0, steps>=1, mixup in (0,1], and an explicit reference spec
    matching the policy's sharding."""
    with pytest.raises(ValueError, match='beta > 0'):
        _check_reference_sync(RLConfig(rl_type='grpo', sync_ref_model=True, beta=0.0), TrainConfig())
    with pytest.raises(ValueError, match='ref_model_sync_steps'):
        _check_reference_sync(RLConfig(rl_type='grpo', sync_ref_model=True, beta=0.04,
                                        ref_model_sync_steps=0), TrainConfig())
    with pytest.raises(ValueError, match='ref_model_mixup_alpha'):
        _check_reference_sync(RLConfig(rl_type='grpo', sync_ref_model=True, beta=0.04,
                                        ref_model_mixup_alpha=0.0), TrainConfig())
    distributed = DistributedConfig(backend='megatron', parallel_spec={'tp': 2})
    with pytest.raises(ValueError, match='parallel_spec'):
        _check_reference_sync(RLConfig(rl_type='grpo', sync_ref_model=True, beta=0.04,
                                       reference_parallel_spec={'tp': 4}), TrainConfig(), distributed)
    _check_reference_sync(RLConfig(rl_type='grpo', sync_ref_model=True, beta=0.04,
                                   reference_parallel_spec={'tp': 2}), TrainConfig(), distributed)


def test_chord_router_replay_and_sampling_replay():
    """S2-34: CHORD's schedule fields must all be explicit and ordered; router/sampling replay
    are GRPO-only; sampling replay forbids a loss-side KL and multi-turn."""
    with pytest.raises(ValueError, match='explicit CHORD schedule fields'):
        _check_chord(RLConfig(rl_type='grpo', chord_sft_dataset='d'))
    with pytest.raises(ValueError, match='chord_mu_valley'):
        _check_chord(RLConfig(rl_type='grpo', chord_sft_dataset='d', chord_sft_per_device_train_batch_size=2,
                              chord_mu_warmup_steps=1, chord_mu_decay_steps=1, chord_mu_peak=0.5,
                              chord_mu_valley=0.9))
    _check_chord(RLConfig(rl_type='grpo', chord_sft_dataset='d', chord_sft_per_device_train_batch_size=2,
                          chord_mu_warmup_steps=1, chord_mu_decay_steps=1, chord_mu_peak=0.8,
                          chord_mu_valley=0.2))

    with pytest.raises(ValueError, match='router_replay_mode'):
        _check_router_replay(RLConfig(rl_type='dpo', router_replay_mode='record'))
    _check_router_replay(RLConfig(rl_type='grpo', router_replay_mode='disabled'))

    with pytest.raises(ValueError, match='enable_sampling_replay is GRPO-only'):
        _check_sampling_replay(RLConfig(rl_type='dpo', enable_sampling_replay=True))
    with pytest.raises(ValueError, match='beta must be 0'):
        _check_sampling_replay(RLConfig(rl_type='grpo', enable_sampling_replay=True, beta=0.04,
                                        kl_in_reward=False, calculate_KL=True))
    with pytest.raises(ValueError, match='multi-turn'):
        _check_sampling_replay(RLConfig(rl_type='grpo', enable_sampling_replay=True, beta=0.0),
                               RolloutConfig(max_turns=4))
    _check_sampling_replay(RLConfig(rl_type='grpo', enable_sampling_replay=True, beta=0.0))


def test_prm_channel_and_scorer_knobs():
    """S2-35: PRM scorer knobs without a --prm channel do nothing (refused); prm_orm_mix is the
    convex-combination ratio in [0,1]; PRM cannot stack with RLSD/SDAR (both per-token)."""
    reward = _explicit(RewardConfig(), 'prm_scorer')
    with pytest.raises(ValueError, match='no --prm channel is set'):
        _check_prm(RLConfig(rl_type='grpo'), reward)

    reward = RewardConfig(prm=['Qwen/PRM-7B'], prm_orm_mix=1.5)
    with pytest.raises(ValueError, match='prm_orm_mix'):
        _check_prm(RLConfig(rl_type='grpo'), reward)

    reward = RewardConfig(prm=['Qwen/PRM-7B'])
    with pytest.raises(ValueError, match='RLSD/SDAR'):
        _check_prm(RLConfig(rl_type='grpo', advantage_reweight='rlsd'), reward)
    reward = RewardConfig(prm=['Qwen/PRM-7B'])
    with pytest.raises(ValueError, match='RLSD/SDAR'):
        _check_prm(RLConfig(rl_type='grpo', sdar_loss_coef=0.1), reward)

    _check_prm(RLConfig(rl_type='grpo'), RewardConfig(prm=['Qwen/PRM-7B'], prm_orm_mix=0.5))


def test_distillation_family_and_self_distill():
    """S2-36: lmbda in [0,1]; offload/deepspeed need a distinct teacher; opsd/mopd refuse a beta
    (no KL term); GKD's JSD knobs are positive; RLSD/SDAR refuse each other and liger; RLSD
    needs the orm channel."""
    with pytest.raises(ValueError, match='lmbda'):
        _check_distillation(RLConfig(rl_type='gkd', lmbda=1.5))
    with pytest.raises(ValueError, match='distinct local teacher_model'):
        _check_distillation(RLConfig(rl_type='gkd', offload_teacher_model=True))
    with pytest.raises(ValueError, match='beta'):
        _check_distillation(RLConfig(rl_type='opsd', beta=0.1))
    with pytest.raises(ValueError, match='sft_alpha'):
        _check_distillation(RLConfig(rl_type='gkd', sft_alpha=-1))
    _check_distillation(RLConfig(rl_type='gkd', lmbda=1.0, sft_alpha=0.1, distill_temperature=1.0))

    from swift.dev.config.validate import _check_rlsd, _check_sdar, _check_self_distillation
    with pytest.raises(ValueError, match='orm channel'):
        _check_rlsd(RLConfig(rl_type='grpo', advantage_reweight='rlsd'))
    with pytest.raises(ValueError, match='cannot be enabled together'):
        _check_sdar(RLConfig(rl_type='grpo', sdar_loss_coef=0.1, advantage_reweight='rlsd'))
    with pytest.raises(ValueError, match='liger'):
        _check_self_distillation(RLConfig(rl_type='grpo', advantage_reweight='rlsd'),
                                 TrainConfig(use_liger_kernel=True), RewardConfig(orm=['accuracy']))


def test_auxiliary_adapters_and_channel_companions():
    """S2-37: ref/teacher adapters take exactly one; teacher_adapters need a teacher; each reward
    channel's companion lists give one entry per MODEL item (rules do not count)."""
    with pytest.raises(ValueError, match='ref_adapters currently supports one'):
        _check_auxiliary_adapters(RLConfig(rl_type='dpo', ref_adapters=['a', 'b']))
    with pytest.raises(ValueError, match='teacher_adapters currently supports one'):
        _check_auxiliary_adapters(RLConfig(rl_type='gkd', teacher_adapters=['a', 'b']))
    with pytest.raises(ValueError, match='teacher_adapters requires'):
        _check_auxiliary_adapters(RLConfig(rl_type='gkd', teacher_adapters=['a']))

    # two model ids in orm, one adapter: misaligned
    reward = RewardConfig(orm=['Qwen/RM-1', 'Qwen/RM-2'], orm_adapters=['only-one'])
    with pytest.raises(ValueError, match='one entry per model id'):
        _check_channel_companions(reward, 'orm')
    # a rule plus a model: companions align with the MODEL subsequence only
    reward = RewardConfig(orm=['accuracy', 'Qwen/RM-1'], orm_adapters=['a'])
    _check_channel_companions(reward, 'orm')
    # companions with no model at all
    reward = RewardConfig(orm=['accuracy'], orm_adapters=['a'])
    with pytest.raises(ValueError, match='requires a model id'):
        _check_channel_companions(reward, 'orm')


# --- rollout / streaming (S2-38..S2-41) ---------------------------------------------------


def test_multi_turn_and_rollout_basics():
    """S2-38: bounds live in validate_multi_turn_config (pinned in the infer suite); here the
    GRPO-only multi-turn gate and the sandbox guard run through validate_rollout_config."""
    with pytest.raises(ValueError, match='GRPO-only'):
        validate_rollout_config(RolloutConfig(max_turns=4), RLConfig(rl_type='dpo'))
    # a hand-built RolloutConfig skips process's in_place derivation, so pin it explicitly
    validate_rollout_config(RolloutConfig(max_turns=4, sampler_mode='colocate',
                                          weight_sync_strategy='in_place'), RLConfig(rl_type='grpo'))
    validate_rollout_config(RolloutConfig(max_turns=4))  # sampling (TUI): exempt
    with pytest.raises(ValueError, match='sandbox_num_envs'):
        validate_rollout_config(RolloutConfig(sandbox_num_envs=0))


def test_resource_mode_and_async_gates(monkeypatch):
    """S2-39: manual/estimate/adaptive; manual refuses inert resource knobs; estimate/adaptive
    are online-only; async is online-only, needs disaggregated, refuses remax/dynamic-sample and
    the off-policy distillation mix; fully_async is grpo/ppo only."""
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    # manual + an explicitly set resource knob: inert, refused
    rollout = _explicit(RolloutConfig(resource_mode='manual'), 'resource_total_gpus')
    rollout.resource_total_gpus = 2
    with pytest.raises(ValueError, match='does not read'):
        validate_rollout_config(rollout, RLConfig(rl_type='grpo'))

    # a non-online recipe with estimate: refused
    rollout = RolloutConfig(resource_mode='estimate')
    with pytest.raises(ValueError, match='grpo/ppo/gkd/opsd/mopd'):
        validate_rollout_config(rollout, RLConfig(rl_type='dpo'))

    # async with a non-online type / colocate / remax / dynamic-sample
    with pytest.raises(ValueError, match='online loops'):
        _check_async_mode(RolloutConfig(async_mode='one_step_off'), RLConfig(rl_type='dpo'))
    grpo = RLConfig(rl_type='grpo', rollout_importance_sampling_mode='token_truncate')
    with pytest.raises(ValueError, match='disaggregated'):
        _check_async_mode(RolloutConfig(async_mode='one_step_off', sampler_mode='colocate'), grpo)
    with pytest.raises(ValueError, match='dynamic_sample'):
        _check_async_mode(RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated'),
                          RLConfig(rl_type='grpo', rollout_importance_sampling_mode='token_truncate',
                                   dynamic_sample=True))
    with pytest.raises(ValueError, match='remax'):
        _check_async_mode(RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated'),
                          RLConfig(rl_type='grpo', rollout_importance_sampling_mode='token_truncate',
                                   advantage_estimator='remax'))
    # grpo async without an IS correction
    with pytest.raises(ValueError, match='off-policy'):
        _check_async_mode(RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated'),
                          RLConfig(rl_type='grpo'))
    # fully_async for distillation: single-step only
    with pytest.raises(ValueError, match='GRPO and PPO'):
        _check_async_mode(RolloutConfig(async_mode='fully_async', sampler_mode='disaggregated'),
                          RLConfig(rl_type='gkd', lmbda=1.0, rollout_importance_sampling_mode='token_truncate'))

    # the streaming-driver helpers the gates lean on
    assert _streams_under_none(RLConfig(rl_type='grpo')) is True
    assert _streams_under_none(RLConfig(rl_type='dpo')) is False
    assert _sync_only_feature(None) is True
    assert _sync_only_feature(RLConfig(rl_type='gkd', lmbda=0.5)) is True
    assert _sync_only_feature(RLConfig(rl_type='gkd', lmbda=1.0)) is False


def test_streaming_publication_pairs():
    """S2-40: parameter_sync_step>=1; one_step_off pins staleness 1; colocate requires in_place;
    adapter_snapshot refuses full-param and partial-rollout; in_place under an overlapping mode
    requires allow_partial_rollout; an unknown strategy is refused."""
    grpo = RLConfig(rl_type='grpo', rollout_importance_sampling_mode='token_truncate')
    rollout = RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated', parameter_sync_step=0)
    with pytest.raises(ValueError, match='parameter_sync_step'):
        _check_streaming_publication(rollout, grpo, None, 'one_step_off')

    rollout = _explicit(RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated',
                                      weight_sync_strategy='adapter_snapshot'), 'max_staleness')
    rollout.max_staleness = 2
    with pytest.raises(ValueError, match='one_step_off'):
        _check_streaming_publication(rollout, grpo, None, 'one_step_off')

    rollout = RolloutConfig(async_mode='none', sampler_mode='colocate', weight_sync_strategy='adapter_snapshot')
    with pytest.raises(ValueError, match='colocate'):
        _check_streaming_publication(rollout, grpo, None, 'none')

    rollout = RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated',
                            weight_sync_strategy='adapter_snapshot', allow_partial_rollout=False)
    with pytest.raises(ValueError, match='no adapter to pin'):
        _check_streaming_publication(rollout, grpo, None, 'one_step_off')

    lora = TunerConfig(tuner='lora')
    rollout = RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated',
                            weight_sync_strategy='adapter_snapshot', allow_partial_rollout=True)
    with pytest.raises(ValueError, match='inert under'):
        _check_streaming_publication(rollout, grpo, lora, 'one_step_off')

    rollout = RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated',
                            weight_sync_strategy='in_place', allow_partial_rollout=False)
    with pytest.raises(ValueError, match='allow_partial_rollout'):
        _check_streaming_publication(rollout, grpo, None, 'one_step_off')

    rollout = RolloutConfig(async_mode='one_step_off', sampler_mode='disaggregated',
                            weight_sync_strategy='in_place', allow_partial_rollout=True)
    _check_streaming_publication(rollout, grpo, None, 'one_step_off')


def test_offload_knobs_and_rollout_sampler(caplog):
    """S2-41: colocate honours offload_model on its own; disaggregated sleep_level warns as inert;
    a rollout engine without weight sync is refused."""
    import logging
    with caplog.at_level(logging.WARNING, logger='swift.dev'):
        rollout = _explicit(RolloutConfig(sampler_mode='colocate'), 'offload_model')
        rollout.offload_model = False
        validate_rollout_config(rollout)
        assert not any('atomic' in rec.message or 'ignoring' in rec.message.lower() for rec in caplog.records)

        caplog.clear()
        rollout = _explicit(RolloutConfig(sampler_mode='disaggregated'), 'sleep_level')
        validate_rollout_config(rollout)
        assert any('sleep_level' in rec.message for rec in caplog.records)

    with pytest.raises(ValueError, match='offload_model'):
        validate_rollout_config(
            RolloutConfig(resource_mode='adaptive', sampler_mode='disaggregated'),
            RLConfig(rl_type='grpo'))

    with pytest.raises(ValueError, match='cannot back an online-RL rollout'):
        validate_rollout_config(RolloutConfig(rollout_sampler='transformers'))


# --- backend / parallel (S2-42..S2-47) -----------------------------------------------------


def test_unsloth_strategy_refusals():
    """S2-42: unsloth rebuilds the graph for whole unsharded weights: deepspeed, fsdp, and
    Ulysses SP are all refused with the plain-backend alternative."""
    with pytest.raises(NotImplementedError, match='DeepSpeed'):
        _check_unsloth_strategy(DistributedConfig(backend='unsloth', deepspeed={'zero_optimization': {}}),
                                TemplateConfig())
    with pytest.raises(NotImplementedError, match='FSDP'):
        _check_unsloth_strategy(DistributedConfig(backend='unsloth', fsdp='full_shard'), TemplateConfig())
    with pytest.raises(NotImplementedError, match='sequence_parallel_size'):
        _check_unsloth_strategy(DistributedConfig(backend='unsloth'),
                                TemplateConfig(sequence_parallel_size=2))
    _check_unsloth_strategy(DistributedConfig(backend='unsloth'), TemplateConfig())


def test_deepspeed_autotp_refused():
    """S2-43: DeepSpeed AutoTP has no TP-aware sampler on the twinkle strategy; refused rather
    than half-wired."""
    with pytest.raises(NotImplementedError, match='AutoTP'):
        _check_deepspeed_autotp(DistributedConfig(deepspeed_autotp_size=2))
    _check_deepspeed_autotp(DistributedConfig())  # unset: no-op


def test_sequence_parallel_tp_needs_tp2():
    """S2-44 (megatron TP-SP half): sequence_parallel shards activations along the TP ranks, so
    TP=1 is refused (legacy silently turned it off)."""
    distributed = DistributedConfig(backend='megatron', sequence_parallel=True, tensor_model_parallel_size=1)
    with pytest.raises(ValueError, match='tensor_model_parallel_size > 1'):
        _check_sequence_parallel_tp(distributed, True)
    distributed.tensor_model_parallel_size = 2
    _check_sequence_parallel_tp(distributed, True)
    _check_sequence_parallel_tp(DistributedConfig(backend='hf', sequence_parallel=True), False)


def test_megatron_fsdp_config_gates():
    """S2-45: megatron fsdp needs the megatron backend and the distributed optimizer, and refuses
    context parallelism (megatron asserts the same)."""
    with pytest.raises(ValueError, match='only applies to the megatron backend'):
        _check_megatron_fsdp(DistributedConfig(backend='hf', use_megatron_fsdp=True), False)
    with pytest.raises(ValueError, match='use_distributed_optimizer'):
        _check_megatron_fsdp(DistributedConfig(backend='megatron', use_megatron_fsdp=True,
                                               use_distributed_optimizer=False), True)
    with pytest.raises(ValueError, match='context_parallel'):
        _check_megatron_fsdp(DistributedConfig(backend='megatron', use_megatron_fsdp=True,
                                               context_parallel_size=2), True)
    _check_megatron_fsdp(DistributedConfig(backend='megatron', use_megatron_fsdp=True), True)


def test_megatron_recompute_and_attn_backend():
    """S2-46: on megatron recompute is driven by recompute_granularity alone -- flag-off with a
    granularity set is a contradiction; padding_free refuses the unfused kernel."""
    with pytest.raises(ValueError, match='contradicts'):
        _check_megatron_recompute(TrainConfig(gradient_checkpointing=False),
                                  DistributedConfig(backend='megatron', recompute_granularity='full'), True)
    _check_megatron_recompute(TrainConfig(gradient_checkpointing=True),
                              DistributedConfig(backend='megatron'), True)  # warn-only path

    pytest.importorskip('megatron.core')
    with pytest.raises(ValueError, match='unfused'):
        _check_megatron_attn_backend(ModelConfig(attn_impl='unfused'), TemplateConfig(padding_free=True), True)
    _check_megatron_attn_backend(ModelConfig(attn_impl='flash'), TemplateConfig(padding_free=True), True)
    _check_megatron_attn_backend(ModelConfig(attn_impl='unfused'), TemplateConfig(padding_free=False), True)


def test_pipeline_layout_and_tp_overlap_gates():
    """S2-47: decoder first/last layer overrides need pp>1; freeze_parameters_ratio refuses
    megatron PP; tp_comm_overlap needs sequence_parallel; selective recompute refuses a method."""
    distributed = DistributedConfig(backend='megatron', decoder_first_pipeline_num_layers=2)
    with pytest.raises(ValueError, match='pipeline_model_parallel_size > 1'):
        _check_pipeline_decoder_layers(distributed, True)
    distributed = DistributedConfig(backend='megatron', pipeline_model_parallel_size=2,
                                    decoder_first_pipeline_num_layers=2)
    _check_pipeline_decoder_layers(distributed, True)

    from swift.dev.config.validate import _check_freeze_ratio_pp
    with pytest.raises(ValueError, match='freeze_parameters_ratio'):
        _check_freeze_ratio_pp(TrainConfig(freeze_parameters_ratio=0.1),
                               DistributedConfig(backend='megatron', pipeline_model_parallel_size=2), True, None)

    with pytest.raises(ValueError, match='sequence_parallel'):
        _check_tp_comm_overlap(DistributedConfig(backend='megatron', tp_comm_overlap=True), True)

    with pytest.raises(ValueError, match='selective'):
        _check_selective_recompute(
            DistributedConfig(backend='megatron', recompute_granularity='selective', recompute_method='uniform'),
            True)


def test_fp4_fp8_quantization_matrix():
    """S2-48 (fp4/fp8 half): format-adjacent knobs need the format; the format is megatron-only;
    param_gather needs the distributed optimizer; fp4 and fp8 are mutually exclusive."""
    model = _explicit(ModelConfig(fp4_recipe='mxfp4'), 'fp4_recipe')
    with pytest.raises(ValueError, match='need ModelConfig.fp4_format'):
        _check_quantization(model, DistributedConfig(backend='megatron'), True)

    model = ModelConfig(fp4_param_gather=True)
    with pytest.raises(ValueError, match='fp4_param_gather'):
        _check_quantization(model, DistributedConfig(backend='megatron'), True)

    model = ModelConfig(fp4_format='mxfp4')
    with pytest.raises(ValueError, match='megatron backend'):
        _check_quantization(model, DistributedConfig(backend='hf'), False)

    model = ModelConfig(fp4_format='mxfp4', fp4_param_gather=True)
    with pytest.raises(ValueError, match='use_distributed_optimizer'):
        _check_quantization(model, DistributedConfig(backend='megatron', use_distributed_optimizer=False), True)

    model = ModelConfig(fp4_format='mxfp4', fp8_format='e4m3')
    with pytest.raises(ValueError, match='mutually exclusive'):
        _check_quantization(model, DistributedConfig(backend='megatron'), True)

    _check_quantization(ModelConfig(fp4_format='mxfp4'), DistributedConfig(backend='megatron'), True)


# --- data / optimizer / checkpoint (S2-50..S2-56) ------------------------------------------


def test_padding_free_needs_flash_attn():
    """S2-50: padding_free / packing on the transformers backend requires a flash kernel; None
    and sdpa are both refused."""
    from swift.dev.config.validate import _check_padding_free_attn
    for attn in (None, 'sdpa', 'eager'):
        with pytest.raises(ValueError, match='flash attention'):
            _check_padding_free_attn(ModelConfig(attn_impl=attn), TemplateConfig(padding_free=True),
                                     DatasetConfig(), False)
    _check_padding_free_attn(ModelConfig(attn_impl='flash_attn'), TemplateConfig(padding_free=True),
                             DatasetConfig(), False)
    _check_padding_free_attn(ModelConfig(attn_impl=None), TemplateConfig(padding_free=True),
                             DatasetConfig(), True)  # the megatron path has its own check


def test_data_knob_conflicts():
    """S2-51: group_by_length refuses padding_free/packing/streaming/lazy; lazy_tokenize refuses
    packing/streaming; data_sharding needs shuffle; streaming refuses a cached_dataset."""
    with pytest.raises(ValueError, match='padding_free'):
        _check_group_by_length(DatasetConfig(group_by_length=True), TemplateConfig(padding_free=True))
    with pytest.raises(ValueError, match='packing'):
        _check_group_by_length(DatasetConfig(group_by_length=True, packing=True), TemplateConfig())
    with pytest.raises(ValueError, match='map-style'):
        _check_group_by_length(DatasetConfig(group_by_length=True, streaming=True), TemplateConfig())
    with pytest.raises(ValueError, match='lazy_tokenize=False'):
        _check_group_by_length(DatasetConfig(group_by_length=True, lazy_tokenize=True), TemplateConfig())

    with pytest.raises(ValueError, match='packing and lazy_tokenize'):
        _check_lazy_tokenize(DatasetConfig(lazy_tokenize=True, packing=True))
    with pytest.raises(ValueError, match='streaming and lazy_tokenize'):
        _check_lazy_tokenize(DatasetConfig(lazy_tokenize=True, streaming=True))
    _check_lazy_tokenize(DatasetConfig(lazy_tokenize=None, packing=True))  # auto backs off to eager

    from swift.dev.config.validate import _check_data_sharding
    with pytest.raises(ValueError, match='train_dataloader_shuffle'):
        _check_data_sharding(DatasetConfig(data_sharding=True, train_dataloader_shuffle=False))

    from swift.dev.config.validate import _check_streaming
    with pytest.raises(ValueError, match='cached_dataset'):
        _check_streaming(DatasetConfig(streaming=True, cached_dataset=['c']))


def test_optimizer_knob_gates():
    """S2-52: GaLore refuses an adapter tuner and the unimplemented legacy knobs; LoRA+ needs a
    lora-family tuner; the megatron-only ``optimizer`` on transformers is refused with the
    pointer to ``optim``; stray muon/qk knobs without optim='muon' are refused."""
    with pytest.raises(ValueError, match='use_galore cannot be combined'):
        _check_galore(TrainConfig(use_galore=True), TunerConfig(tuner='lora'))
    train = _explicit(TrainConfig(use_galore=True), 'galore_with_embedding')
    train.galore_with_embedding = True
    with pytest.raises(NotImplementedError, match='GaLore extensions'):
        _check_galore(train, None)
    _check_galore(TrainConfig(use_galore=True, optim='adamw'), None)

    with pytest.raises(ValueError, match='lorap_lr_ratio'):
        _check_param_groups(TunerConfig(tuner='trainable_tokens', lorap_lr_ratio=0.1))
    _check_param_groups(TunerConfig(tuner='lora', lorap_lr_ratio=0.1))

    with pytest.raises(ValueError, match='Megatron optimizer'):
        _check_muon(TrainConfig(optimizer='muon'), DistributedConfig(backend='hf'), False)
    train = _explicit(TrainConfig(), 'muon_momentum')
    train.muon_momentum = 0.95
    with pytest.raises(ValueError, match='configure the muon optimizer'):
        _check_muon(train, DistributedConfig(backend='hf'), False)
    _check_muon(TrainConfig(optim='muon', muon_momentum=0.95), DistributedConfig(backend='hf'), False)


def test_checkpoint_runtime_gates():
    """S2-53: save_strategy must be steps; resume_only_model=false without a resume point would
    silently start from scratch; async_save refuses push_to_hub; transformers refuses the
    megatron-only selective save/load knobs; megatron pins safetensors; ppo needs limit>=2."""
    training = {'training': True}

    with pytest.raises(NotImplementedError, match='save_strategy'):
        _check_checkpoint_runtime(CheckpointConfig(save_strategy='epoch'), DistributedConfig(), None, None,
                                  False, **training)

    checkpoint = _explicit(CheckpointConfig(resume_only_model=False), 'resume_only_model')
    with pytest.raises(ValueError, match='silently start from scratch'):
        _check_checkpoint_runtime(checkpoint, DistributedConfig(), None, None, False, **training)

    with pytest.raises(ValueError, match='push_to_hub'):
        _check_checkpoint_runtime(CheckpointConfig(async_save=True, push_to_hub=True), DistributedConfig(),
                                  None, None, False, **training)

    checkpoint = _explicit(CheckpointConfig(no_save_rng=True), 'no_save_rng')
    with pytest.raises(NotImplementedError, match='selectively omit'):
        _check_checkpoint_runtime(checkpoint, DistributedConfig(backend='hf'), None, None, False, **training)

    with pytest.raises(NotImplementedError, match='safetensors'):
        _check_checkpoint_runtime(CheckpointConfig(save_safetensors=False),
                                  DistributedConfig(backend='megatron'), None, None, True, **training)

    with pytest.raises(ValueError, match='save_total_limit >= 2'):
        _check_checkpoint_runtime(CheckpointConfig(save_total_limit=1), DistributedConfig(backend='hf'), None,
                                  RLConfig(rl_type='ppo'), False, **training)
    _check_checkpoint_runtime(CheckpointConfig(), DistributedConfig(backend='hf'), None, None, False, **training)


def test_save_total_limit_bounds():
    """S2-54: limit < 1 refused; async_save conflicts with the rolling delete (cannot tell whether
    a background write landed); megatron keeps legacy's >= 2 bound."""
    with pytest.raises(ValueError, match='>= 1'):
        _check_save_total_limit(CheckpointConfig(save_total_limit=0), False)
    with pytest.raises(ValueError, match='async_save'):
        _check_save_total_limit(CheckpointConfig(save_total_limit=3, async_save=True), False)
    with pytest.raises(ValueError, match='>= 2'):
        _check_save_total_limit(CheckpointConfig(save_total_limit=1), True)
    _check_save_total_limit(CheckpointConfig(save_total_limit=2), True)
    _check_save_total_limit(CheckpointConfig(), False)  # None: unlimited, fine


def test_rlhf_reference_and_layout_gates():
    """S2-55: a ref model passed to an algorithm that never loads one is refused; ref_adapters
    without a ref model is refused; padding_free/packing is wired for most RLHF types (ppo/opsd/
    mopd refused with the reason); RL requires right padding (logp extraction)."""
    with pytest.raises(ValueError, match='ref_adapters requires'):
        _check_rlhf_ref_model(ModelConfig(), None, RLConfig(rl_type='dpo', ref_adapters=['a']))
    with pytest.raises(ValueError, match='not used by rl_type'):
        _check_rlhf_ref_model(ModelConfig(), TunerConfig(tuner='lora'),
                              RLConfig(rl_type='cpo', ref_model='m'))
    _check_rlhf_ref_model(ModelConfig(), None, RLConfig(rl_type='dpo', ref_model='m'))

    for rl_type, message in (('ppo', 'value-head critic'), ('opsd', 'teacher->student'),
                             ('mopd', 'K teacher logps')):
        with pytest.raises(ValueError, match=message):
            _check_rlhf_padding_free(TemplateConfig(padding_free=True), DatasetConfig(),
                                     RLConfig(rl_type=rl_type))
    _check_rlhf_padding_free(TemplateConfig(padding_free=True), DatasetConfig(), RLConfig(rl_type='grpo'))

    with pytest.raises(ValueError, match='padding_side="right"'):
        _check_rlhf_padding_side(TemplateConfig(padding_side='left'), RLConfig(rl_type='dpo'))
    _check_rlhf_padding_side(TemplateConfig(padding_side='right'), RLConfig(rl_type='dpo'))


def test_eval_and_logging_gates(monkeypatch):
    """S2-56: predict_with_generate needs an eval_dataset; the transformers eval sampler refuses
    SP; vllm/sglang eval needs ray; eval_iters is -1 or positive; report_to refuses unknown
    trackers and 'none' mixed with an active one; swanlab email needs its fields."""
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.setenv('WORLD_SIZE', '1')
    with pytest.raises(ValueError, match='eval_dataset'):
        _check_eval_generation(TrainConfig(predict_with_generate=True), TemplateConfig(), DistributedConfig())

    with pytest.raises(ValueError, match='sequence_parallel'):
        _check_eval_generation(TrainConfig(predict_with_generate=True, eval_dataset=['arc']),
                               TemplateConfig(sequence_parallel_size=2), DistributedConfig())
    with pytest.raises(NotImplementedError, match='mode'):
        _check_eval_generation(TrainConfig(predict_with_generate=True, eval_dataset=['arc'],
                                           eval_sampler_backend='vllm'), TemplateConfig(),
                               DistributedConfig(mode='local'))

    with pytest.raises(ValueError, match='eval_iters'):
        _check_eval_iters(TrainConfig(eval_iters=0))
    _check_eval_iters(TrainConfig(eval_iters=-1))
    _check_eval_iters(TrainConfig(eval_iters=5))

    with pytest.raises(ValueError, match='Unsupported.*report_to'):
        _check_logging(LoggingConfig(report_to=['mlflow']))
    with pytest.raises(ValueError, match='cannot combine'):
        _check_logging(LoggingConfig(report_to=['none', 'tensorboard']))
    with pytest.raises(ValueError, match='SwanLab email'):
        _check_logging(LoggingConfig(logging_strategy='steps', logging_steps=5,
                                     swanlab_notification_method='email'))
    _check_logging(LoggingConfig(report_to=['tensorboard'], logging_steps=5))


def test_infer_store_and_multi_turn_surface():
    """S2-61 (infer half): a non-incremental store_format refuses resume (the arrow-format half
    is pinned in the infer suite); validate_multi_turn passes a sane surface."""
    from swift.dev.config import InferConfig
    with pytest.raises(ValueError, match='cannot be combined with resume'):
        validate_infer_config(InferConfig(store_format='arrow', resume=True))
    validate_infer_config(InferConfig(store_format='jsonl', resume=True))
    validate_multi_turn_config(RolloutConfig(max_turns=4, max_trajectory_tokens=1024))
# --- orchestration / dispatch (S2-57, S2-59, S2-61 convert, S2-62) -------------------------

import os
import tempfile

from swift.dev.config.validate import _changed_fields, _check_backend_specific, validate_convert_config


def test_sft_full_orchestration_smoke(tmp_path):
    """S2-57 (L1-anchored): process_and_validate_configs runs every process derivation and every
    validate check over a parsed sft config set -- the last gate before a real run. A local model
    directory keeps the hub off; no accelerator is touched (resolve stops before weights)."""
    from swift.dev.cli.sft import parse_sft_configs
    from swift.dev.config import process_and_validate_configs

    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    configs = parse_sft_configs([
        '--model', str(model_dir), '--load_args', 'false', '--dataset', 'd',
        '--output_dir', str(tmp_path / 'out'), '--logging_strategy', 'steps', '--logging_steps', '5',
    ])
    process_and_validate_configs(configs)
    # the orchestration wrote its versioned output dir -- read it back from disk, not memory
    output = configs['checkpoint_config'].output_dir
    assert os.path.isdir(output) and str(tmp_path / 'out') in output

    # and the orchestration really runs validate_configs: a backend-illegal knob in the same
    # argv shape must be refused HERE, not silently pass into the run
    configs = parse_sft_configs([
        '--model', str(model_dir), '--load_args', 'false', '--dataset', 'd',
        '--output_dir', str(tmp_path / 'out2'), '--logging_strategy', 'steps', '--logging_steps', '5',
        '--tensor_model_parallel_size', '2',
    ])
    with pytest.raises(ValueError, match='only implemented by the megatron backend'):
        process_and_validate_configs(configs)


def test_changed_fields_reports_only_user_touched():
    """S2-59: with no _explicit_fields the diff-vs-defaults path lists what differs; once the
    parser records explicit fields, ONLY those are reported -- an explicit default still counts
    (the user chose it), while defaults the user never named never fire refusal gates."""
    assert _changed_fields(None) == []

    # parsed Configs always carry _explicit_fields (an empty set until the CLI sets one): the
    # empty set reports nothing, and once a field was spelled on the CLI ONLY those are reported.
    fresh = TrainConfig()
    assert _changed_fields(fresh) == []
    explicit_default = TrainConfig()
    explicit_default._explicit_fields = {'learning_rate'}  # an explicit --learning_rate 1e-5
    assert _changed_fields(explicit_default) == ['learning_rate']

    # the diff-vs-defaults path is for holders without _explicit_fields (None): anything that
    # differs from the default is reported -- the parser never relies on it, but the guards can.
    touched = TrainConfig(learning_rate=3e-5)  # 3e-5, NOT the 1e-5 default: a real diff
    touched._explicit_fields = None
    assert 'learning_rate' in _changed_fields(touched)
    pristine = TrainConfig()
    pristine._explicit_fields = None
    assert 'learning_rate' not in _changed_fields(pristine)


def test_convert_config_validates_only_writable_stores():
    """S2-61 (convert half): None or a run without to_cached_dataset skips the store check; the
    export-side store must be a REGISTERED and WRITABLE format (bin/orc/lance/builder are
    read-only sources or reserved phases)."""
    from swift.dev.config import ConvertConfig
    validate_convert_config(None)
    validate_convert_config(ConvertConfig())
    validate_convert_config(ConvertConfig(to_cached_dataset=True, store_format='jsonl'))

    with pytest.raises(ValueError, match='unknown store_format'):
        validate_convert_config(ConvertConfig(to_cached_dataset=True, store_format='parquet'))
    with pytest.raises(ValueError, match='cannot be written'):
        validate_convert_config(ConvertConfig(to_cached_dataset=True, store_format='bin'))


@pytest.mark.parametrize('backend', ['hf', 'megatron'])
def test_backend_specific_dispatch_refuses_the_wrong_side(backend):
    """S2-62: _check_backend_specific dispatches on the backend -- a transformers run refuses
    _MEGATRON_ONLY knobs (e.g. a megatron parallel size), a megatron run refuses _HF_ONLY knobs
    (e.g. deepspeed); a None tuner_config skips its tuner-only entries; all-off defaults pass."""
    is_megatron = backend == 'megatron'
    model_config = ModelConfig(model='m')
    dataset_config = DatasetConfig(dataset=['d'])
    train_config = TrainConfig()
    distributed_config = DistributedConfig(backend=backend)

    # the wrong-backend knob, explicitly set
    if is_megatron:
        distributed_config = DistributedConfig(backend=backend, deepspeed={'zero_optimization': {}})
        expected = 'only implemented by the transformers backend'
    else:
        distributed_config = DistributedConfig(backend=backend, pipeline_model_parallel_size=2)
        expected = 'only implemented by the megatron backend'
    with pytest.raises(ValueError, match=expected):
        _check_backend_specific(model_config, dataset_config, train_config, distributed_config, is_megatron)

    # all-off defaults pass on both backends; a None tuner_config never trips its tuner-only rows
    _check_backend_specific(ModelConfig(model='m'), DatasetConfig(dataset=['d']), TrainConfig(),
                            DistributedConfig(backend=backend), is_megatron, tuner_config=None)

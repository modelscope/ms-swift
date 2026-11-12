# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

# The f-divergences DPO can stand in for the KL; validate derives its accepted set from this alias
# via typing.get_args, so the static type and the validator cannot drift apart.
F_DIVERGENCE_TYPE = Literal['reverse_kl', 'forward_kl', 'js_divergence', 'alpha_divergence']


@dataclass
class RLConfig:
    """RLHF algorithm hyperparameters for all supported rl_type variants."""

    # === Core ===
    # Which algorithm the RL command runs; opsd/mopd/rft are this refactor's first-class additions.
    rl_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd', 'opsd', 'mopd',
                     'rft'] = 'dpo'
    ref_model: Optional[str] = None
    ref_adapters: List[str] = field(default_factory=list)
    ref_model_type: Optional[str] = None
    ref_model_revision: Optional[str] = None
    # Parallel layout for the reference model's DeviceGroup ('dp2'/'tp2'...); None = single rank
    # (card-frugal but pins a large reference to one GPU). A LoRA/disable_adapter reference ignores it.
    ref_parallel_spec: Optional[str] = None
    beta: Optional[float] = None
    max_completion_length: int = 512

    # === DPO ===
    label_smoothing: float = 0
    rpo_alpha: Optional[float] = None
    ld_alpha: Optional[float] = None
    discopop_tau: float = 0.05
    loss_type: Optional[List[str]] = None
    loss_weights: Optional[List[float]] = None
    cpo_alpha: float = 1.0
    simpo_gamma: float = 1.0

    # === KTO ===
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0

    # === PPO ===
    num_ppo_epochs: int = 4
    whiten_rewards: bool = False
    kl_coef: float = 0.05
    cliprange: float = 0.2
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95

    # === GRPO ===
    num_generations: int = 8
    log_completions: bool = False
    num_iterations: int = 1
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    delta: Optional[float] = None
    advantage_estimator: Literal[
        'grpo', 'rloo', 'reinforce_plus_plus', 'reinforce_plus_plus_baseline', 'remax', 'opo', 'gpg', 'grpo_passk'
    ] = 'grpo'
    f_norm: float = 1.0  # GPG advantage normalization divisor; 1.0 disables
    kl_in_reward: Optional[bool] = None
    scale_rewards: Optional[Literal['group', 'batch', 'none', 'gdpo']] = None
    importance_sampling_level: Literal['token', 'sequence', 'sequence_token'] = 'token'
    dynamic_sample: bool = False
    max_resample_times: int = 3
    overlong_filter: bool = False

    # === Additional policy-loss knobs (clip_cov / kl_cov / dppo / dro) ===
    dro_beta: float = 0.1  # DRO quadratic log-ratio penalty strength (loss_type='dro'); > 0
    # DPPO truncated IS weight bound; the divergence threshold reuses epsilon/epsilon_high (verl).
    clip_ratio_c: float = 20.0
    clip_cov_ratio: float = 0.0002  # Clip-Cov fraction of valid tokens zeroed out
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002  # KL-Cov fraction of valid tokens with the extra KL penalty
    ppo_kl_coef: float = 1.0

    # === CHORD auxiliary SFT ===
    chord_sft_dataset: List[str] = field(default_factory=list)
    chord_sft_per_device_train_batch_size: Optional[int] = None
    chord_enable_phi_function: bool = False
    chord_mu_warmup_steps: Optional[int] = None
    chord_mu_decay_steps: Optional[int] = None
    chord_mu_peak: Optional[float] = None
    chord_mu_valley: Optional[float] = None

    # === Reference synchronization ===
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 512
    ref_model_mixup_alpha: float = 0.6
    # Parallel layout for the reference model; None inherits the policy's parallel_spec -- the
    # paired EMA sync needs identical sharding, so same-spec is the correct default.
    reference_parallel_spec: Optional[str] = None
    log_entropy: bool = False
    top_entropy_quantile: float = 1.0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    fipo_decay_rate: float = 32.0
    fipo_clip_range: Optional[float] = 0.2
    fipo_clip_high_only: bool = True
    fipo_safety_threshold: Optional[float] = 4.0
    teacher_kl_coef: float = 1.0

    # === RLSD / SDAR self-distillation ===
    advantage_reweight: Optional[Literal['rlsd']] = None
    rlsd_lambda: float = 0.5
    rlsd_reweight_clip_range: float = 0.2
    rlsd_lambda_warmup_steps: int = 0
    rlsd_lambda_decay_steps: int = 0
    rlsd_negative_only: bool = False
    sdar_loss_coef: float = 0.0
    sdar_gate_beta: float = 5.0

    rollout_importance_sampling_mode: Optional[Literal['token_truncate', 'token_mask', 'sequence_truncate',
                                                       'sequence_mask']] = None
    rollout_importance_sampling_threshold: float = 2.0
    log_rollout_offpolicy_metrics: bool = False
    off_policy_sequence_mask_delta: Optional[float] = None

    # === GKD ===
    sft_alpha: float = 0
    lmbda: float = 0.5
    gkd_logits_topk: Optional[int] = None
    # DISTILLATION temperature: softens teacher/student distributions and scales logits before
    # sampled-token log-probs; NOT the rollout temperature (see sampling_temperature).
    distill_temperature: float = 0.9
    # SAMPLING temperature for the student's on-policy generation; None falls back to
    # distill_temperature (the pre-split behavior).
    sampling_temperature: Optional[float] = None

    # === RM ===
    center_rewards_coefficient: Optional[float] = None

    # === Teacher Model ===
    # Always a list; GKD/OPSD distil from exactly ONE (assembly rejects more), MOPD reads all.
    teacher_model: Optional[List[str]] = None
    teacher_adapters: List[str] = field(default_factory=list)
    teacher_model_type: Optional[str] = None
    teacher_model_revision: Optional[str] = None
    teacher_deepspeed: Optional[str] = None
    offload_teacher_model: bool = False
    # Per-teacher fusion weights for MOPD, positional with teacher_model; normalized to sum 1 in
    # MOPDLoss (length mismatch fails at loss time). None = 1/K each.
    teacher_weights: Optional[List[float]] = None
    # Parallel layout for the teacher DeviceGroup (gkd/opsd/mopd); None = single rank, serial.
    teacher_parallel_spec: Optional[str] = None
    # How MOPD's K teachers are PLACED: 'pack' (default) shares one group (card-frugal, serial);
    # 'spread' gives each its own group (concurrent at K x the GPUs).
    auxiliary_placement: Literal['pack', 'spread'] = 'pack'
    # OPSD divergence direction (OPSDLoss.reverse); True = the paper's teacher - student.
    opsd_reverse: bool = True

    # === RFT / RAFT / ReST (rejection-sampling fine-tuning) ===
    # Completions sampled per prompt each iteration; None reuses num_generations.
    rft_num_samples: Optional[int] = None
    # How sampled completions are filtered: 'best_of_n' keeps the best per prompt, 'threshold'
    # keeps >= rft_threshold, 'top_k' keeps the rft_top_k best. Empty selection fails loudly.
    rft_select: Literal['best_of_n', 'threshold', 'top_k'] = 'best_of_n'
    rft_threshold: float = 0.0  # reward cutoff for rft_select='threshold'
    rft_top_k: int = 1  # per-prompt keep count for rft_select='top_k'
    # Rejection-sampling rounds (sample -> score -> filter -> SFT -> re-sync); 1 = one best-of-n pass.
    rft_iterations: int = 1
    # Hard cap on kept completions per prompt (a memory bound for 'threshold'); None keeps all.
    rft_max_samples_per_prompt: Optional[int] = None

    # Compute the KL term explicitly rather than folding it into the advantage; None = per-algorithm.
    calculate_KL: Optional[bool] = None
    f_divergence_type: F_DIVERGENCE_TYPE = 'reverse_kl'
    f_alpha_divergence_coef: float = 0.5  # only read when f_divergence_type='alpha_divergence'
    # Drop the reference model and score against a constant: removes a whole model from memory and
    # the anchor keeping the policy near where it started.
    reference_free: bool = False
    real_tau: float = 0.5  # temperature on the REAL objective's soft constraint
    # Replay the router's expert choices from the generating pass in the training pass, so an MoE
    # policy's log-probs are computed under the routing that produced the tokens.
    router_replay_mode: Literal['disabled', 'R2', 'R3'] = 'disabled'
    # Replay the sampler's per-token distribution in the training forward so the GRPO ratio is
    # measured against the exact support set (GRPO-only; no KL penalty, no multi-turn).
    enable_sampling_replay: bool = False
    # Teacher outputs obtained by disabling the policy's adapter instead of loading a second model;
    # only valid when the teacher is exactly the base model of a LoRA policy. Private.
    _teacher_use_disable_adapter: bool = False

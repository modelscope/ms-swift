"""RLHF algorithm hyperparameters (DPO/KTO/CPO/PPO/GRPO/GKD/RM)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# TODO: integrate it
@dataclass
class RLHFConfig:
    """RLHF algorithm hyperparameters for all supported rlhf_type variants."""

    # === Core ===
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd'] = 'dpo'
    ref_model: Optional[str] = None
    ref_adapters: List[str] = field(default_factory=list)
    ref_model_type: Optional[str] = None
    ref_model_revision: Optional[str] = None
    beta: Optional[float] = None
    max_completion_length: int = 512
    loss_scale: Optional[str] = None

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
    num_mini_batches: int = 1
    local_rollout_forward_batch_size: int = 64
    num_sample_generations: int = 10
    missing_eos_penalty: Optional[float] = None

    # === GRPO ===
    num_generations: int = 8
    reward_funcs: List[str] = field(default_factory=list)
    reward_weights: Optional[List[float]] = None
    log_completions: bool = False
    num_iterations: int = 1
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    delta: Optional[float] = None
    advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo'
    kl_in_reward: Optional[bool] = None
    scale_rewards: Optional[Literal['group', 'batch', 'none', 'gdpo']] = None
    importance_sampling_level: Literal['token', 'sequence', 'sequence_token'] = 'token'
    dynamic_sample: bool = False
    max_resample_times: int = 3
    overlong_filter: bool = False
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 512
    ref_model_mixup_alpha: float = 0.6
    log_entropy: bool = False
    top_entropy_quantile: float = 1.0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    fipo_decay_rate: float = 32.0
    fipo_clip_range: Optional[float] = 0.2
    fipo_clip_high_only: bool = True
    fipo_safety_threshold: Optional[float] = 4.0
    teacher_kl_coef: float = 1.0
    rollout_importance_sampling_mode: Optional[Literal['token_truncate', 'token_mask', 'sequence_truncate',
                                                       'sequence_mask']] = None
    rollout_importance_sampling_threshold: float = 2.0
    log_rollout_offpolicy_metrics: bool = False
    off_policy_sequence_mask_delta: Optional[float] = None

    # === GRPO Reward Function Parameters ===
    cosine_min_len_value_wrong: float = -0.5
    cosine_max_len_value_wrong: float = 0.0
    cosine_min_len_value_correct: float = 1.0
    cosine_max_len_value_correct: float = 0.5
    cosine_max_len: Optional[int] = None
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0
    soft_max_length: Optional[int] = None
    soft_cache_length: Optional[int] = None

    # === GRPO Multi-turn ===
    multi_turn_scheduler: Optional[str] = None
    max_turns: Optional[int] = None
    completion_length_limit_scope: Literal['total', 'per_round'] = 'per_round'
    use_gym_env: Optional[bool] = None
    gym_env: Optional[str] = None

    # === GKD ===
    sft_alpha: float = 0
    lmbda: float = 0.5
    gkd_logits_topk: Optional[int] = None
    temperature: float = 0.9

    # === RM ===
    center_rewards_coefficient: Optional[float] = None

    # === Teacher Model ===
    teacher_model: Optional[str] = None
    teacher_adapters: List[str] = field(default_factory=list)
    teacher_model_type: Optional[str] = None
    teacher_model_revision: Optional[str] = None
    teacher_deepspeed: Optional[str] = None
    teacher_model_server: Optional[str] = None
    offload_teacher_model: bool = False

    # === Reward Model ===
    reward_model: Optional[List[str]] = None
    reward_adapters: List[str] = field(default_factory=list)
    reward_model_type: Optional[List[str]] = None
    reward_model_revision: Optional[List[str]] = None
    reward_model_plugin: Optional[List[str]] = None

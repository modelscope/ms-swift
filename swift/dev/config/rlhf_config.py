"""RLHF algorithm hyperparameters (DPO/KTO/CPO/PPO/GRPO/GKD/OPSD/MOPD/RFT/RM)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional


@dataclass
class RLHFConfig:
    """RLHF algorithm hyperparameters for all supported rlhf_type variants."""

    # === Core ===
    #: Which algorithm the RL command runs. ``opsd``/``mopd``/``rft`` are the three first-class methods
    #: added this refactor (self-distillation, multi-teacher distillation, rejection-sampling FT); each
    #: dispatches to its own recipe in ``run_rlhf`` and reuses the fields grouped below.
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd', 'opsd', 'mopd',
                       'rft'] = 'dpo'
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

    # === Reward channels (one selector shared by GRPO and best-of-n synthesis) ===
    #: Outcome-reward channel. Each item is a registered rule name (an ``orms`` key), a reward plugin
    #: class/callable, or a reward-model id -- the merge of the old ``reward_funcs`` (rules) and infer's
    #: ``orm_model`` (a model) into one list, so a channel mixes rules and a model in the order typed and
    #: ``orm_weights`` aligns to that order. GRPO reads ``orm`` as rules only (its reward models come from
    #: ``reward_model`` below); best-of-n synthesis resolves a non-rule item through the reward-model path.
    orm: List[Any] = field(default_factory=list)
    #: Per-item weights for ``orm``; ``None`` weights every item equally. Length must match the resolved
    #: channel (rules + model), enforced when the per-item scores are combined.
    orm_weights: Optional[List[float]] = None
    #: Parallel layout for this channel's reward model when ``orm`` names a model id (a scalar RM or an
    #: independent generative judge) -- e.g. ``dp2``, ``tp2``. It is the reward model's OWN layout, apart
    #: from the run's ``--parallel_spec`` (the policy / sampler's): it sizes the channel's dedicated
    #: DeviceGroup and drives the model's DeviceMesh (scalar RM) or the judge engine's
    #: ``tensor_parallel_size`` (independent judge). ``None`` keeps the old behavior -- the channel
    #: inherits ``nproc_per_node`` as a pure data-parallel replica set. A scalar RM rides the transformers
    #: backend, which does not shard weights by tp/pp, so its spec expresses dp/fsdp/ep/ulysses; tp/pp need
    #: the megatron backend or a generative judge. Rule items in the channel hold no model and ignore this.
    orm_parallel_spec: Optional[str] = None
    #: Process-reward channel, with the same heterogeneous item grammar as ``orm``. Two consumers: best-of-n
    #: synthesis ranks with it as the second axis, and GRPO turns it into a per-token *process* reward -- each
    #: reasoning step is scored by its response prefix (see ``prm_scorer``) and that score is broadcast onto
    #: the step's tokens, placed on the intermediate tokens while the outcome (``orm``) reward lands on the
    #: last one (segmented placement). Rule items score the whole response as one scalar; a model id is a
    #: frozen PRM built/placed exactly like an ``orm`` model.
    prm: List[Any] = field(default_factory=list)
    #: Per-item weights for ``prm``; ``None`` weights every item equally.
    prm_weights: Optional[List[float]] = None
    #: Parallel layout for this channel's reward model, exactly as ``orm_parallel_spec`` but for ``prm``.
    prm_parallel_spec: Optional[str] = None
    #: Which ``prm_scorer`` plugin segments a response into reasoning steps and broadcasts each step's PRM
    #: score onto its tokens (GRPO process reward only). A registered name, a class, or a callable resolved
    #: through :meth:`swift.dev.plugin.PluginRegistry.resolve`; ``None`` uses the default ``'delimiter'``
    #: scorer (:class:`swift.dev.rewards.prm.DelimiterPRMScorer`).
    prm_scorer: Optional[Any] = None
    #: Step boundary the default ``'delimiter'`` PRM scorer splits the decoded response on -- one reasoning
    #: step per delimiter-separated segment. A newline (one step per line) by default; a scorer that does not
    #: segment by text ignores this. ``None`` means the default newline.
    prm_step_delimiter: Optional[str] = None

    # === GRPO ===
    num_generations: int = 8
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
    # Multi-turn is enabled by setting max_turns. Per-turn length is sampling_params.max_tokens;
    # max_trajectory_tokens caps the whole trajectory (both are independent knobs, either may be None).
    max_turns: Optional[int] = None
    max_trajectory_tokens: Optional[int] = None

    # === GKD ===
    sft_alpha: float = 0
    lmbda: float = 0.5
    gkd_logits_topk: Optional[int] = None
    #: DISTILLATION (logit-scoring) temperature -- softens the teacher/student distributions inside the
    #: divergence (GKD's β-JSD; for OPSD/MOPD it scales the logits before the sampled-token log-probs are
    #: taken). This is the model-forward / loss knob, NOT how hot the student rollout is sampled. They were
    #: one field before (W16), which forced the generation temperature to equal the scoring temperature; see
    #: ``sampling_temperature`` for the now-separate rollout knob.
    temperature: float = 0.9
    #: SAMPLING temperature for the student's on-policy generation (GKD/OPSD/MOPD in-process rollouts).
    #: Kept apart from the distillation ``temperature`` so a run can, e.g., sample greedily-ish (low) while
    #: scoring softly (high). ``None`` preserves the pre-split behavior by falling back to ``temperature``,
    #: so existing runs are unchanged; set it to decouple the two.
    sampling_temperature: Optional[float] = None

    # === RM ===
    center_rewards_coefficient: Optional[float] = None

    # === Teacher Model ===
    #: The frozen teacher(s), always a list so the CLI can pass several space-separated ids (a bare
    #: ``Union[str, List[str]]`` is parsed single-valued -- see cli/parser.py's ``str_list_none`` case -- so
    #: MOPD's K teachers could not be expressed). GKD/OPSD and GRPO's RLSD/SDAR distil from exactly ONE
    #: entry (``recipe.assembly.single_teacher_id`` rejects more, pointing at MOPD); MOPD reads the whole
    #: list, one frozen model per domain teacher, each scored with ``forward_only``. OPSD's headline setting
    #: needs NO external teacher -- it distills the policy's own base via ``disable_lora`` (see
    #: ``_teacher_use_disable_adapter``) -- so ``teacher_model=None`` is valid there. There is deliberately
    #: no HTTP teacher server (no-server / all-Ray premise, plan §2.H).
    teacher_model: Optional[List[str]] = None
    teacher_adapters: List[str] = field(default_factory=list)
    teacher_model_type: Optional[str] = None
    teacher_model_revision: Optional[str] = None
    teacher_deepspeed: Optional[str] = None
    offload_teacher_model: bool = False
    #: Per-teacher fusion weights for MOPD (``rlhf_type='mopd'``), positional with a list-valued
    #: ``teacher_model``. Non-negative; normalized to sum 1 inside ``MOPDLoss``. ``None`` weights every
    #: teacher equally (``1/K``). Length must equal the teacher count -- enforced at loss time, so a
    #: mismatch fails loudly rather than silently dropping or padding a channel. Ignored by single-teacher
    #: methods (GKD/OPSD), which take the one teacher at full weight.
    teacher_weights: Optional[List[float]] = None
    #: Parallel layout for the distillation teacher DeviceGroup (gkd/opsd/mopd), same grammar as
    #: ``orm_parallel_spec`` (e.g. ``dp2``, ``tp2``). ``None`` sizes the group to a single rank (serial
    #: ``forward_only``, card-frugal -- the stage-one default); a spec gives the frozen teacher(s) their own
    #: sizing for a big or concurrency-friendly teacher. MOPD's K teachers share this one group. It is the
    #: teachers' OWN layout, apart from the run's ``--parallel_spec`` (the policy's).
    teacher_parallel_spec: Optional[str] = None
    #: OPSD divergence direction, passed straight to ``OPSDLoss.reverse``. ``True`` (default) is the
    #: paper's direction ``r = teacher_logp - student_logp`` -- teacher-preferred tokens pull the student
    #: up. ``False`` swaps it (``r = student_logp - teacher_logp``) for a quick ablation without touching
    #: the loss. Only read by OPSD/MOPD; other types ignore it.
    opsd_reverse: bool = True

    # === RFT / RAFT / ReST (rejection-sampling fine-tuning) ===
    #: Completions sampled per prompt each iteration. ``None`` reuses ``num_generations`` so an RFT run
    #: shares GRPO's rollout sizing by default; set it to override independently.
    rft_num_samples: Optional[int] = None
    #: How the sampled completions are filtered down to the SFT set. ``best_of_n`` keeps the single
    #: highest-reward completion per prompt (RAFT's default, D4-A); ``threshold`` keeps every completion
    #: scoring ``>= rft_threshold``; ``top_k`` keeps the ``rft_top_k`` best per prompt. An empty selected
    #: set fails loudly (never a silent 0-step run).
    rft_select: Literal['best_of_n', 'threshold', 'top_k'] = 'best_of_n'
    #: Reward cutoff for ``rft_select='threshold'``. Ignored by the other two selectors.
    rft_threshold: float = 0.0
    #: Per-prompt keep count for ``rft_select='top_k'``. Ignored by the other two selectors.
    rft_top_k: int = 1
    #: Rejection-sampling rounds: sample -> score -> filter -> SFT -> re-sync the updated policy into the
    #: sampler, repeated this many times (ReST's iterative loop). ``1`` is a single best-of-n SFT pass.
    rft_iterations: int = 1
    #: Optional hard cap on kept completions per prompt after selection (a memory/throughput bound for
    #: ``threshold``, which can otherwise keep an unbounded number). ``None`` keeps everything selected.
    rft_max_samples_per_prompt: Optional[int] = None

    # === Reward Model ===
    reward_model: Optional[List[str]] = None
    reward_adapters: List[str] = field(default_factory=list)
    reward_model_type: Optional[List[str]] = None
    reward_model_revision: Optional[List[str]] = None
    #: Chat template per reward model, positional with ``reward_model``. Needed because a reward model
    #: is often trained under a different template than the policy, and scoring under the wrong one
    #: silently changes what it rewards. None lets each model use its own default.
    reward_template: Optional[List[str]] = None

    # === Megatron backend ===
    # The Megatron path's counterparts to the fields above. Kept separate rather than folded in because
    # a reference model in mcore format is not interchangeable with ``ref_model``: it is sharded under
    # Megatron parameter names, so the two are loaded by different code.
    #: Reference model in mcore format, and an mcore LoRA to apply to it.
    mcore_ref_model: Optional[str] = None
    mcore_ref_adapter: Optional[str] = None
    #: Compute the KL term explicitly rather than folding it into the advantage. None takes the value
    #: implied by the algorithm.
    calculate_KL: Optional[bool] = None
    #: Which f-divergence stands in for the KL, e.g. 'reverse_kl', 'forward_kl', 'js_divergence'.
    f_divergence_type: str = 'reverse_kl'
    #: Coefficient of the 'alpha_divergence' f-divergence; only read when ``f_divergence_type='alpha_divergence'``.
    f_alpha_divergence_coef: float = 0.5
    #: Drop the reference model entirely and score against a constant instead. Removes a whole model
    #: from memory, and with it the anchor that keeps the policy near where it started.
    reference_free: bool = False
    #: Temperature on the REAL objective's soft constraint.
    real_tau: float = 0.5
    #: Replay the router's expert choices from the generating pass during the training pass, so an MoE
    #: policy's log-probabilities are computed under the routing that actually produced the tokens.
    #: 'disabled' recomputes routing, which can silently make the importance ratio wrong.
    router_replay_mode: Literal['disabled', 'R2', 'R3'] = 'disabled'
    #: Replay the sampler's per-token sampling distribution during the training forward, so the GRPO
    #: importance ratio is measured against the exact support set each token was drawn from instead of a
    #: recomputed full-vocab softmax. Needs the vLLM sampler to export sampling masks (dev turns on the
    #: engine's ``enable_sampling_replay``, which sets vLLM ``enable_return_sampling_mask`` +
    #: ``processed_logprobs``). twinkle's GRPOLoss forbids it alongside a KL penalty (``beta``) or entropy
    #: bonus, and the forward forbids it under sequence/context parallelism; GRPO-only. Distinct from
    #: ``rollout_importance_sampling_mode`` (a truncated-IS correction on the recomputed ratio).
    enable_sampling_replay: bool = False
    #: Obtain the teacher's outputs by disabling the policy's adapter instead of loading a second model.
    #: Only valid when the teacher is exactly the base model of a LoRA policy. Private: it is set from
    #: the teacher configuration above rather than passed directly.
    _teacher_use_disable_adapter: bool = False

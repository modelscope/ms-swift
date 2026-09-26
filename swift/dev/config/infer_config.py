"""Batch-inference configuration: which engine runs, and what the run writes out.

This single config drives both offline infer shapes: plain batch inference over a dataset (one
response per row plus the acc/rouge metrics) and best-of-n synthesis (generate a candidate group per
prompt, score it with reward functions, then write DPO-shaped rows with optional rollout-token
sidecars and checkpointed resume). The synthesis-only knobs live in the clearly-marked sections below;
a plain infer run simply leaves them at their defaults.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class InferConfig:
    """Engine choice, result handling, and best-of-n synthesis knobs for an offline inference run.

    Deliberately separate from GenerationConfig, which says how to sample, and from RolloutConfig,
    whose ``vllm_*`` / ``sglang_*`` fields tune a chosen engine: this is which engine to choose, what
    to do with its output, and (for synthesis) how many candidates to keep and how to score them. A
    run switches backends by changing ``infer_backend`` alone. Reward-function selection is NOT here --
    it reuses ``RLHFConfig.reward_funcs`` / ``reward_weights``, the same registry surface GRPO uses.
    """

    # === Engine ===
    #: 'pt' is a compatibility alias of 'transformers'. lmdeploy is intentionally outside v5.
    #: 'client' distils from a remote OpenAI-compatible teacher (no local weights); 'no' is the
    #: message-only companion -- both are synthesis-only backends a plain infer run never selects.
    infer_backend: Literal['vllm', 'transformers', 'sglang', 'pt', 'client', 'no'] = 'transformers'
    #: Requests batched into one engine call. Only meaningful for the transformers backend, since vLLM
    #: and SGLang do their own continuous batching and ignore it.
    max_batch_size: int = 1

    # === Dataset ===
    #: Take only this many rows from the validation set. None runs all of them.
    val_dataset_sample: Optional[int] = None

    # === Output ===
    #: jsonl file the completions are appended to. None writes under the run's output directory.
    result_path: Optional[str] = None
    #: How sampled candidates become output rows. 'all' writes one row per prompt carrying every
    #: completion in ``responses`` (plus per-candidate ``scores`` when a reward func scored them) --
    #: plain inference and scored-corpus dumping alike. 'dpo' scores the group and writes best-of-n
    #: chosen/rejected pairs for DPO training (needs ``num_return_sequences >= 2``). Scoring is
    #: orthogonal to this knob: a reward func set on an 'all' run still scores every candidate and
    #: stores its score. A future 'grpo' would write the whole scored group with group-relative stats.
    output_format: Literal['all', 'dpo'] = 'all'

    # === Scoring ===
    #: Metric computed over the completions once the run finishes. None only generates.
    metric: Optional[Literal['acc', 'rouge']] = None
    #: Apply the activation to reranker logits, turning them into scores comparable across queries.
    #: Only consulted by reranker models.
    reranker_use_activation: bool = True

    # === Synthesis: mode and engine ===
    #: 'sample' scores locally-generated candidates; 'distill' draws them from a remote teacher
    #: (``infer_backend='client'``). This is a label only -- it does not change the sampling call (driven
    #: by ``infer_backend``) and no longer derives any other field. Kept for the legacy ``--sampler_type``
    #: spelling.
    sampler_type: Literal['sample', 'distill'] = 'sample'
    #: Extra keyword arguments merged into the engine args, for any backend -- a plain inference run tunes
    #: its engine the same way a synthesis run does.
    engine_kwargs: Optional[Dict[str, Any]] = None

    # === Synthesis: candidates ===
    #: Completions generated per prompt. Plain inference wants 1; raise this (or pass the CLI's
    #: --num_samples) to draw a best-of-n group for reward ranking / 'dpo' output.
    num_return_sequences: int = 1
    #: How many top-scoring candidates become positives per prompt. The lowest scorer is the
    #: rejected_response, so n_best_to_keep < num_return_sequences or there is no negative left.
    n_best_to_keep: int = 5

    # === Synthesis: reward models and judge ===
    #: Frozen reward-model channels. A ``*_model`` that is an http(s) URL routes to the API judge;
    #: otherwise a scalar (seq_cls) RM is built locally, or -- when the base is the sampler's own model
    #: -- a generative judge reuses the sampling engine. ``*_adapter`` is a frozen LoRA on the channel.
    orm_model: Optional[str] = None
    prm_model: Optional[str] = None
    orm_adapter: Optional[str] = None
    prm_adapter: Optional[str] = None
    #: Override for the built-in generative-judge prompt: a ``str.format`` template rendered with
    #: ``prompt=`` and ``completion=``, asked to emit a single number. Only used by a generative judge.
    judge_template: Optional[str] = None

    # === Synthesis: reward knobs ===
    #: Candidates scoring at or below this are dropped. None keeps every candidate.
    reward_threshold: Optional[float] = None
    #: Legacy sampling spelling of ``reward_threshold``: the infer CLI folds it into ``reward_threshold``
    #: when the latter is not given explicitly. Kept for old sampling configs/scripts -- prefer
    #: ``reward_threshold``, the one field the ranking actually reads.
    prm_threshold: Optional[float] = None
    #: Drop the whole prompt when this fraction of its candidates already score above
    #: ``reward_threshold`` -- an easy prompt teaches the model nothing. None keeps every prompt.
    easy_query_threshold: Optional[float] = None
    #: Min-max normalise each prompt's scores into [0, 1] before ranking and thresholding, so
    #: ``reward_threshold`` means "relative to this prompt's own group" rather than an absolute value.
    normalize_rewards: bool = False
    #: Score the reference answer alongside the candidates and include it in the ranking; with
    #: per-group normalisation the ground truth is the anchor that defines what 1.0 means.
    score_ground_truth: bool = False
    #: The second reward channel, scored by ``prm_funcs`` and combined with the ORM channel.
    prm_funcs: List[Any] = field(default_factory=list)
    prm_weights: Optional[List[float]] = None
    #: Multiplier on the ORM channel when both channels are in play (legacy's hard-coded 10), so an
    #: existing recipe can reproduce legacy ranking exactly.
    orm_channel_weight: float = 1.0

    # === Batching ===
    #: Prompts per sampler call, and the granularity at which results are flushed (and, when resuming,
    #: checkpointed) -- a crash loses at most one batch. Distinct from ``max_batch_size`` above, which is
    #: the transformers engine's per-call request count.
    batch_size: int = 1
    #: Stop after this many batches. None runs the whole dataset.
    max_batches: Optional[int] = None
    #: Legacy spellings; the CLI folds these into batch_size/max_batches when explicitly provided.
    num_sampling_batch_size: Optional[int] = None
    num_sampling_batches: Optional[int] = None

    # === Synthesis: output and resume ===
    #: Persist each kept candidate's rollout tokens (input_ids / labels / logprobs / loss mask) to an NPZ
    #: sidecar beside the jsonl, and embed the relative path plus the reward score in the row. Off by
    #: default: it needs a token-capable local backend (not the message-only ``client`` teacher).
    save_rollout_tokens: bool = False
    #: Continue a previous run from its checkpoint instead of starting over.
    resume: bool = False
    #: Overwrite an existing complete output file instead of returning early.
    override_exist_file: bool = False
    #: Previously-produced jsonl files whose rows are reused instead of resampled, keyed by prompt.
    #: Distinct from ``resume``: these are OTHER runs' outputs -- the way to add candidates to a corpus,
    #: or to re-score without paying for generation.
    cache_files: List[str] = field(default_factory=list)
    #: Tolerate per-row failures during generation (transformers backend only). Off by default so a
    #: systematic problem surfaces on row one instead of quietly thinning the whole output.
    strict: bool = True

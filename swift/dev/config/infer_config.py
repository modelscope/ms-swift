# Copyright (c) ModelScope Contributors. All rights reserved.
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
    run switches backends by changing ``sampler`` alone. Reward-channel selection is NOT here --
    it reuses ``RLConfig.orm`` / ``prm`` (+ their ``*_weights``), the same registry surface GRPO uses.
    """

    # === Engine ===
    #: Which sampler runs the model: a built-in engine name ('vllm' / 'transformers' / 'sglang', with 'pt'
    #: a compatibility alias of 'transformers'; lmdeploy is intentionally outside v5), a twinkle ``Sampler``
    #: subclass, or an external plugin source (a local file/folder or a ``hf://`` / ``ms://`` id) resolved
    #: through the unified loader -- see ``swift.dev.builders.sampler._resolve_sampler``. One message-only
    #: value builds no local sampler: 'no', the cache-only backend (every prompt must be covered by
    #: ``cache_files``) -- a synthesis-only backend a plain infer run never selects.
    sampler: Any = 'transformers'
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
    #: chosen/rejected pairs for DPO training (needs ``num_return_sequences >= 2``). 'grpo' writes one row
    #: per prompt holding the whole sampled group as an offline GRPO corpus: every candidate's full
    #: trajectory plus its rollout tokens -- input_ids / labels / completion_mask / old_logps, via
    #: ``save_rollout_tokens``, which 'grpo' requires -- and each candidate's ``scores`` when a reward
    #: channel scored the group. The group-relative advantage is NOT computed here; training recomputes it
    #: from the stored rewards + logps. Needs ``num_return_sequences >= 2``. Scoring is orthogonal to this
    #: knob: a reward func set on an 'all' run still scores every candidate and stores its score.
    output_format: Literal['all', 'dpo', 'grpo'] = 'all'
    #: Container the result is written in, orthogonal to ``output_format`` above (that is the row shape,
    #: this is the serialisation). An OPEN registry key, not a closed vocabulary: any format registered in
    #: ``swift.dev.dataset.storage.STORAGE_MAPPING`` is accepted -- the built-in ``arrow``/``jsonl``/``bin``
    #: plus anything a ``@register_storage`` subclass adds (loaded via ``--external_plugins``). Shared with
    #: ``export --to_cached_dataset``. ``jsonl`` (default) keeps the line-delimited dump with its resumable
    #: checkpoint writers and, under ``save_rollout_tokens``, the per-candidate NPZ sidecar; it is the only
    #: incrementally-resumable format. Every other writable format (``arrow``, a custom one) serialises the
    #: whole run once at the end through the shared store writer, so it is rejected together with ``resume``.
    #: An unregistered or read-only name is refused at config validation (``validate_store_format``), before
    #: any generation, rather than after the run has produced rows it cannot write.
    store_format: str = 'jsonl'
    #: Restrict the persisted columns to this allow-list; None keeps every column the emit produced. A
    #: name that matches no column is rejected rather than silently dropped.
    store_fields: Optional[List[str]] = None

    # === Scoring ===
    #: Metric computed over the completions once the run finishes. None only generates.
    metric: Optional[Literal['acc', 'rouge']] = None
    #: Apply the activation to reranker logits, turning them into scores comparable across queries.
    #: Only consulted by reranker models.
    reranker_use_activation: bool = True

    # === Synthesis: engine ===
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
    #: Reward-channel selection (model ids, LoRA adapters, templates, rule hyperparameters) lives on
    #: ``RewardConfig`` -- the ``--orm``/``--prm`` lists and their companion families, shared with
    #: online RL so both businesses speak one selector. Only the synthesis-side knobs stay here.

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
    #: Multiplier on the ORM channel when both channels are in play (legacy's hard-coded 10), so an
    #: existing recipe can reproduce legacy ranking exactly. The ORM/PRM channel selectors and their
    #: per-item weights live on RLConfig (``orm`` / ``prm`` / ``orm_weights`` / ``prm_weights``).
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
    #: default: it needs a token-capable local backend (not the message-only ``no`` backend).
    save_rollout_tokens: bool = False
    #: Continue a previous run from its checkpoint instead of starting over.
    resume: bool = False
    #: Overwrite an existing complete output file instead of returning early.
    override_exist_file: bool = False
    #: Previously-produced run outputs whose rows are reused instead of resampled, keyed by prompt.
    #: Any store the storage layer reads back qualifies (jsonl, an arrow store directory, ...) -- one
    #: run's ``result_path`` can serve as another run's cache verbatim. Distinct from ``resume``: these
    #: are OTHER runs' outputs -- the way to add candidates to a corpus, or to re-score without paying
    #: for generation.
    cache_files: List[str] = field(default_factory=list)
    #: Raise on a per-row generation failure (transformers backend only) rather than tolerating it. True by
    #: default so a systematic problem surfaces on row one instead of quietly thinning the whole output;
    #: set False to skip failed rows. Named apart from ``DatasetConfig.strict`` (the column-mapping
    #: tolerance) so ``--strict`` keeps a single unambiguous owner across commands.
    generation_strict: bool = True

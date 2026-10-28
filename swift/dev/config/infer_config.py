# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class InferConfig:
    """Engine choice, result handling, and best-of-n synthesis knobs for an offline inference run.

    Reward channels live on RewardConfig (orm/prm + companions), not fields of their own here.
    """

    # === Engine ===
    # 'vllm'/'transformers'/'sglang' ('pt' alias), a Sampler subclass/plugin source, 'no' (none),
    # or 'twinkle': the client-side sampler over a twinkle-server (needs --mode remote +
    # --server_base_url; model names the SERVER-side served model).
    sampler: Any = 'transformers'
    # Requests batched into one engine call; transformers backend only (vLLM/SGLang self-batch).
    max_batch_size: int = 1

    # === Dataset ===
    val_dataset_sample: Optional[int] = None  # take only this many rows; None runs all

    # === Output ===
    # jsonl file the completions are appended to; None writes under the run's output directory.
    result_path: Optional[str] = None
    # 'all' = every completion per prompt; 'dpo' = best-of-n pairs; 'grpo' = whole sampled group +
    # rollout tokens (needs save_rollout_tokens and num_return_sequences >= 2).
    output_format: Literal['all', 'dpo', 'grpo'] = 'all'
    # OPEN registry key (dataset.storage.STORAGE_MAPPING built-ins + @register_storage plugins);
    # jsonl is the only incrementally-resumable format.
    store_format: str = 'jsonl'
    # Allow-list of persisted columns; None keeps every column the emit produced.
    store_fields: Optional[List[str]] = None

    # === Scoring ===
    metric: Optional[Literal['acc', 'rouge']] = None  # None only generates
    # Activation on reranker logits; only consulted by reranker models.
    reranker_use_activation: bool = True

    # === Synthesis: engine ===
    # Extra kwargs merged into the engine args, for any backend.
    engine_kwargs: Optional[Dict[str, Any]] = None

    # === Synthesis: candidates ===
    # Completions generated per prompt; 1 for plain inference, raise it for a best-of-n group.
    num_return_sequences: int = 1
    # Top-scoring candidates kept as positives per prompt; the lowest scorer is the rejected one.
    n_best_to_keep: int = 5

    # === Synthesis: reward models and judge ===
    # Reward-channel selection (model ids, LoRA, templates, rule hyperparameters) lives on
    # RewardConfig: the --orm/--prm lists and their companion families, shared with online RL.

    # === Synthesis: reward knobs ===
    reward_threshold: Optional[float] = None  # candidates scoring at/below this are dropped
    # Legacy sampling spelling of reward_threshold, folded into it when not given explicitly.
    prm_threshold: Optional[float] = None
    # Drop the whole prompt when this fraction of its candidates already score above threshold.
    easy_query_threshold: Optional[float] = None
    # Min-max normalise each prompt's scores before ranking/thresholding (relative, not absolute).
    normalize_rewards: bool = False
    # Score the reference answer too and include it in the ranking (the anchor of 1.0).
    score_ground_truth: bool = False
    # Multiplier on the ORM channel when both are in play (legacy's hard-coded 10).
    orm_channel_weight: float = 1.0

    # === Batching ===
    # Prompts per sampler call and the flush granularity; distinct from max_batch_size.
    batch_size: int = 1
    max_batches: Optional[int] = None  # stop after this many; None runs the whole dataset
    # Legacy spellings; the CLI folds these into batch_size/max_batches when explicitly provided.
    num_sampling_batch_size: Optional[int] = None
    num_sampling_batches: Optional[int] = None

    # === Synthesis: output and resume ===
    # Rollout tokens to an NPZ sidecar beside the jsonl; needs a token-capable local backend.
    save_rollout_tokens: bool = False
    resume: bool = False  # continue a previous run from its checkpoint
    override_exist_file: bool = False  # overwrite an existing complete output instead of returning early
    # Other runs' outputs reused instead of resampled, keyed by prompt (distinct from resume: add
    # candidates / re-score for free). Any readable store qualifies: jsonl, an arrow dir, ...
    cache_files: List[str] = field(default_factory=list)
    # Raise on a per-row generation failure (transformers backend) rather than tolerating it;
    # named apart from DatasetConfig.strict (column-mapping tolerance) so --strict has one owner.
    generation_strict: bool = True

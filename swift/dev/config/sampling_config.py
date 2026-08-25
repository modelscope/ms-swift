"""Best-of-n sampling: how many candidates, how they are scored, and how a run resumes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SamplingConfig:
    """Knobs for ``run_sampling`` -- candidate generation, reward filtering, checkpointed resume.

    Scoring is deliberately NOT a sampler concern: ``reward_funcs`` names entries in swift's ``orms``
    registry (all pure-Python) or passes callables, and is resolved by ``swift.dev.reward``. Nothing
    here loads a model, which is why the sampling backend and the scoring path are independent.
    """

    # === Candidates ===
    #: Completions generated per prompt. The group the reward ranking then sorts.
    num_return_sequences: int = 4
    #: How many top-scoring candidates become positives per prompt. The lowest scorer is the
    #: rejected_response, so n_best_to_keep < num_return_sequences or there is no negative left.
    n_best_to_keep: int = 1

    # === Reward ===
    #: Registered ``orms`` names and/or callables, resolved via ``swift.dev.reward.get_reward_funcs``.
    #: Empty means no filtering: every candidate is emitted as a positive (plain sampling).
    reward_funcs: List[Any] = field(default_factory=list)
    #: Per-func weights for the combined score; defaults to all ones.
    reward_weights: Optional[List[float]] = None
    #: Hyperparameter carrier for the ORMs that read one. Registered ORMs are constructed as
    #: ``orms[name](args=reward_config)``, and some reach into it for their own fields --
    #: ``cosine`` wants ``cosine_min_len_value_wrong`` and friends, ``repetition`` wants
    #: ``repetition_n_grams``. This config is NOT that object (it has no such fields, so passing
    #: itself would raise AttributeError); supply the carrier explicitly, or leave it None when every
    #: chosen ORM is parameter-free (accuracy / format / math / react_format).
    reward_config: Optional[Any] = None
    #: Candidates scoring at or below this are dropped. None keeps every candidate.
    reward_threshold: Optional[float] = None
    #: Drop the whole prompt when this fraction of its candidates already score above
    #: ``reward_threshold`` -- an easy prompt teaches the model nothing. None keeps every prompt.
    easy_query_threshold: Optional[float] = None
    #: Min-max normalise each prompt's scores into [0, 1] before ranking and thresholding.
    #: Makes ``reward_threshold`` mean "relative to this prompt's own group" rather than an absolute
    #: value, which is what legacy did -- useful when reward magnitudes vary wildly between prompts,
    #: misleading when they are already comparable (normalising then hides that a whole group is bad).
    normalize_rewards: bool = False
    #: Score the reference answer alongside the candidates and include it in the ranking. Legacy always
    #: did this: with per-group normalisation the ground truth acts as the anchor that defines what 1.0
    #: means. Without normalisation it mostly just adds a guaranteed-good positive.
    score_ground_truth: bool = False
    #: Weights for the second reward channel, scored by ``prm_funcs``. Legacy kept ORM and PRM apart
    #: and combined them as ``prm + orm * 10``; with explicit weights that 10x is stated rather than
    #: hidden, and either channel may be empty.
    prm_funcs: List[Any] = field(default_factory=list)
    prm_weights: Optional[List[float]] = None
    #: Multiplier applied to the ORM channel when both channels are in play, i.e. legacy's hard-coded
    #: 10. It exists so an existing recipe can reproduce legacy's ranking exactly.
    orm_channel_weight: float = 1.0

    # === Batching ===
    #: Prompts per sampler call. Also the checkpoint granularity: a crash loses at most one batch.
    batch_size: int = 16
    #: Stop after this many batches. None runs the whole dataset.
    max_batches: Optional[int] = None
    #: ``(index, total)`` -- take only piece ``index`` of the dataset, for splitting one dataset
    #: across independent processes/machines. There is no cross-piece coordination.
    data_range: Optional[tuple] = None

    # === Output & resume ===
    output_file: str = 'sampled.jsonl'
    #: Continue a previous run from its checkpoint instead of starting over.
    resume: bool = False
    #: Overwrite an existing complete output file instead of returning early.
    override_exist_file: bool = False
    #: Previously-produced jsonl files whose rows are reused instead of resampled, keyed by prompt.
    #: Distinct from ``resume``: resume continues THIS run's checkpoint, whereas these are other runs'
    #: outputs -- the way to add candidates to a corpus, or to re-score without paying for generation.
    cache_files: List[str] = field(default_factory=list)
    #: Tolerate per-row failures during generation (transformers backend only). Off by default so a
    #: systematic problem surfaces on row one instead of quietly thinning the whole output.
    strict: bool = True

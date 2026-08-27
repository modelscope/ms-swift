"""Batch-inference configuration: which engine runs, and what the run writes out."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class InferConfig:
    """Engine choice and result handling for an inference run.

    Deliberately separate from GenerationConfig, which says how to sample, and from RolloutConfig,
    whose ``vllm_*`` / ``sglang_*`` fields tune a chosen engine: this is only which engine to choose
    and what to do with its output. A run switches backends by changing ``infer_backend`` alone.
    """

    # === Engine ===
    #: 'pt' is an alias of 'transformers'. 'lmdeploy' is accepted for legacy command lines but is not
    #: a migrated backend -- prefer 'vllm' or 'sglang'.
    infer_backend: Literal['vllm', 'transformers', 'sglang', 'lmdeploy', 'pt'] = 'transformers'
    #: Requests batched into one engine call. Only meaningful for the transformers backend, since vLLM
    #: and SGLang do their own continuous batching and ignore it.
    max_batch_size: int = 1

    # === Dataset ===
    #: Take only this many rows from the validation set. None runs all of them.
    val_dataset_sample: Optional[int] = None

    # === Output ===
    #: jsonl file the completions are appended to. None writes under the run's output directory.
    result_path: Optional[str] = None
    #: Rows buffered before a flush. Large enough that writing is not the bottleneck, small enough that
    #: an interrupted run keeps most of its work.
    write_batch_size: int = 1000

    # === Scoring ===
    #: Metric computed over the completions once the run finishes. None only generates.
    metric: Optional[Literal['acc', 'rouge']] = None
    #: Apply the activation to reranker logits, turning them into scores comparable across queries.
    #: Only consulted by reranker models.
    reranker_use_activation: bool = True

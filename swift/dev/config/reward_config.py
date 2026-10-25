# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class RewardConfig:
    """The --orm/--prm reward channels, their models, and the rule plugins' hyperparameters.

    One shared surface for online RL (GRPO/PPO/RFT) and best-of-n synthesis, so neither business
    borrows the other's Config; rule plugins read their knobs off it (cls(args=config)).
    """

    # === Channels ===
    # Each item: a rule name, a plugin class/callable, or a reward-model id; weights align to the
    # typed order. A non-rule string is a frozen MODEL id (scalar-only online; judge on synthesis).
    orm: List[Any] = field(default_factory=list)
    orm_weights: Optional[List[float]] = None  # per-item weights; None = equal
    # LoRA for each MODEL id in orm (positional with the model subsequence). One entry with no model
    # id = reuse the sampler itself as a generative judge wearing that LoRA (synthesis path only).
    orm_adapters: Optional[List[str]] = None
    # model_type / model_revision / chat template per MODEL id in orm (positional, subsequence).
    orm_model_type: Optional[List[str]] = None
    # Task head to load each orm MODEL id as (positional, subsequence). Overrides the checkpoint's own
    # task_type inference -- set 'seq_cls' to load a plain causal-LM checkpoint as a scalar (num_labels=1)
    # reward model; without it a causal-LM id resolves to a generative judge, not a scalar RM.
    orm_task_type: Optional[List[str]] = None
    orm_model_revision: Optional[List[str]] = None
    # A reward model often trains under a different template than the policy; wrong one = silent drift.
    orm_template: Optional[List[str]] = None
    # Parallel layout for this channel's reward model; sizes the channel's dedicated DeviceGroup
    # (None = inherit nproc_per_node as pure DP replicas).
    orm_parallel_spec: Optional[str] = None
    # Process-reward channel, same grammar as orm: synthesis ranks with it as a second axis; GRPO
    # turns it into a per-token process reward blended by prm_orm_mix.
    prm: List[Any] = field(default_factory=list)
    prm_weights: Optional[List[float]] = None
    # LoRA for the MODEL id in prm, positional as orm_adapters; no model id = sampler-reuse judge.
    prm_adapters: Optional[List[str]] = None
    # model_type / model_revision / chat template for the MODEL id in prm (positional, as orm's).
    prm_model_type: Optional[List[str]] = None
    # Task head for the prm MODEL id, as orm_task_type (online RL places only a SCALAR frozen PRM).
    prm_task_type: Optional[List[str]] = None
    prm_model_revision: Optional[List[str]] = None
    prm_template: Optional[List[str]] = None
    prm_parallel_spec: Optional[str] = None
    # Which prm_scorer plugin segments a response into steps; None picks by rollout shape ('turn'
    # when max_turns is set, 'delimiter' otherwise).
    prm_scorer: Optional[Any] = None
    # Step boundary for the 'delimiter' scorer; None = newline. Ignored by non-text scorers.
    prm_step_delimiter: Optional[str] = None
    # PRM/ORM fusion on every response token: mix*process + (1-mix)*orm; 0 = standard GRPO, 1 = pure
    # process. Must be in [0, 1]; only read when a --prm channel is active for rl_type=grpo.
    prm_orm_mix: float = 0.5
    # str.format judge prompt (prompt=/completion=); generative/api judges only, shared by RL and
    # synthesis so a judge renders verdicts identically wherever it scores.
    judge_template: Optional[str] = None

    # === Reward rule hyperparameters ===
    # Read by the cosine / repetition / soft-length rules off this config (cls(args=config)).
    cosine_min_len_value_wrong: float = -0.5
    cosine_max_len_value_wrong: float = 0.0
    cosine_min_len_value_correct: float = 1.0
    cosine_max_len_value_correct: float = 0.5
    cosine_max_len: Optional[int] = None
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0
    soft_max_length: Optional[int] = None
    soft_cache_length: Optional[int] = None

    # === Code Execution Reward (--orm code_execution) ===
    # Scores a generated program against the dataset's test_cases column; read off this config.
    code_language: str = 'python'
    code_timeout: int = 10
    code_max_concurrency: int = 8
    code_memory_limit_gb: Optional[float] = 2.0
    code_sandbox_template: Optional[str] = None

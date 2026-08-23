"""Model loading, architecture, and precision configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


@dataclass
class ModelConfig:
    """Model path, architecture, dtype, and device mapping."""

    model: Optional[str] = None
    model_type: Optional[str] = None
    model_revision: Optional[str] = None
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker', None] = None
    torch_dtype: Literal['bfloat16', 'float16', 'float32', None] = None
    attn_impl: Optional[str] = None
    experts_impl: Optional[str] = None
    new_special_tokens: List[str] = field(default_factory=list)
    num_labels: Optional[int] = None
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification', None] = None
    rope_scaling: Optional[str] = None
    max_model_len: Optional[int] = None
    device_map: Optional[Union[dict, str]] = None
    max_memory: Optional[Union[dict, str]] = None
    local_repo_path: Optional[str] = None
    init_strategy: Literal['zero', 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                           'kaiming_normal', 'orthogonal', None] = None

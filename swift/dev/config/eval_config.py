"""Evaluation harness configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class EvalConfig:
    """EvalScope backend, datasets, output, and remote-service selection."""

    eval_dataset: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None
    eval_dataset_args: Dict[str, Any] = field(default_factory=dict)
    eval_generation_config: Dict[str, Any] = field(default_factory=dict)
    eval_output_dir: str = 'eval_output'
    eval_backend: Literal['Native', 'OpenCompass', 'VLMEvalKit'] = 'Native'
    local_dataset: bool = False
    eval_num_proc: int = 16
    extra_eval_args: Dict[str, Any] = field(default_factory=dict)
    eval_url: Optional[str] = None
    result_jsonl: Optional[str] = None
    use_chat_template: bool = True

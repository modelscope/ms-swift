"""Evaluation harness configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalConfig:
    """EvalScope Native datasets, generation, output, and result selection.

    Evaluation runs in-process off a twinkle sampler through EvalScope's Native runner, so there is no
    remote-service URL, no backend choice (Native only), and no chat-template toggle here -- chat formatting
    is the template's job (``TemplateConfig.use_chat_template``).
    """

    eval_dataset: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None
    eval_dataset_args: Dict[str, Any] = field(default_factory=dict)
    eval_generation_config: Dict[str, Any] = field(default_factory=dict)
    eval_output_dir: str = 'eval_output'
    eval_num_proc: int = 16
    extra_eval_args: Dict[str, Any] = field(default_factory=dict)
    result_jsonl: Optional[str] = None

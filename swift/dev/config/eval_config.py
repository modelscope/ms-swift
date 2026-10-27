# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalConfig:
    """EvalScope Native datasets, generation, output, and result selection.

    Runs in-process off a twinkle sampler through EvalScope's Native runner; with
    ``eval_url`` set, scores an already-served OpenAI-compatible endpoint instead
    and nothing is loaded locally.
    """

    eval_dataset: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None
    eval_dataset_args: Dict[str, Any] = field(default_factory=dict)
    eval_generation_config: Dict[str, Any] = field(default_factory=dict)
    eval_output_dir: str = 'eval_output'
    eval_num_proc: int = 16
    extra_eval_args: Dict[str, Any] = field(default_factory=dict)
    result_jsonl: Optional[str] = None
    #: OpenAI-compatible base_url to score (e.g. http://127.0.0.1:8100/v1); ``--model``
    #: is then the served model name, and no local engine is built.
    eval_url: Optional[str] = None
    #: Bearer key for ``eval_url`` endpoints that require one; local deployments omit it.
    eval_api_key: Optional[str] = None

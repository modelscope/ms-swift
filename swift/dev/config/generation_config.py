# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GenerationConfig:
    """Sampling, beam search, streaming, and structured output settings."""

    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1
    stream: Optional[bool] = None
    stop_words: List[str] = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    structured_outputs_regex: Optional[str] = None

    # A generation_config.json path/dir/instance supplying the baseline the fields above override;
    # named after HF's argument so existing command lines keep working.
    generation_config: Optional[Any] = None

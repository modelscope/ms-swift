"""Generation parameters for inference and deployment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


# TODO: integrate it
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

    # === Defaults from a file ===
    #: A ``generation_config.json``, a directory holding one, or a ready GenerationConfig. Supplies the
    #: baseline that the fields above then override, which is how a model's shipped sampling defaults
    #: are honoured without restating them. Named after HF's argument despite sitting inside a class of
    #: the same name -- renaming it would break every command line that already passes it.
    generation_config: Optional[Any] = None

    # === Seq2seq evaluation ===
    # Read by the trainer during predict/evaluate, not by a plain generate() call: they exist because
    # evaluation needs a fixed budget even when the fields above are left unset.
    generation_max_length: Optional[int] = None
    generation_num_beams: Optional[int] = None

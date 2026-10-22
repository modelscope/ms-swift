# Copyright (c) ModelScope Contributors. All rights reserved.
"""What an encode is FOR, expressed as immutable call data instead of a shared ``template.mode``.

Three orthogonal rendering decisions:

* ``add_generation_prompt`` -- open assistant turn (rollout / inference) vs complete conversation;
* ``with_labels`` -- emit next-token ``labels`` (and loss-side fields);
* ``expand_media`` -- expand media placeholders to model token count vs leave collapsed for engine.

Whether a sample is a preference pair is decided by its type (see ``inputs.SampleInput``), not the intent.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

from .inputs import TemplateInputs

EngineBackend = Literal["vllm", "sglang", "lmdeploy", "transformers"]


@dataclass(frozen=True)
class EncodeIntent:
    add_generation_prompt: bool
    with_labels: bool
    expand_media: bool

    def __post_init__(self):
        if self.with_labels and self.add_generation_prompt:
            raise ValueError(
                "EncodeIntent(with_labels=True, add_generation_prompt=True) is contradictory: an open "
                "generation prompt has no response to supervise."
            )
        if not self.expand_media and self.with_labels:
            raise ValueError(
                "EncodeIntent(expand_media=False, with_labels=True) is contradictory: labels are aligned "
                "with the expanded sequence the model forward consumes."
            )

    def render_mode(self, engine: EngineBackend = "vllm") -> str:
        if self.with_labels:
            return "train"
        if not self.expand_media:
            return engine
        return "transformers"


TRAIN = EncodeIntent(add_generation_prompt=False, with_labels=True, expand_media=True)
INFER = EncodeIntent(add_generation_prompt=True, with_labels=False, expand_media=True)
ENGINE = EncodeIntent(add_generation_prompt=True, with_labels=False, expand_media=False)
SCORE = EncodeIntent(add_generation_prompt=False, with_labels=False, expand_media=True)


def resolve_intent(intent: Optional[EncodeIntent], add_generation_prompt: Optional[bool]) -> EncodeIntent:
    if intent is None:
        return INFER if add_generation_prompt else TRAIN
    if add_generation_prompt is not None and add_generation_prompt != intent.add_generation_prompt:
        raise ValueError(f"add_generation_prompt={add_generation_prompt} contradicts {intent}.")
    return intent


def check_generation_prompt(sample: Union[Dict[str, Any], TemplateInputs], add_generation_prompt: bool) -> None:
    messages = sample.chosen.messages if isinstance(sample, TemplateInputs) else sample.get("messages")
    if not messages:
        return
    last = messages[-1]
    # Tool-call messages are a completed assistant turn in either message format.
    role = last.get("role")
    open_turn = role not in {"assistant", "tool_call"} or (
        role == "assistant" and last.get("content") is None and not last.get("tool_calls")
    )
    if add_generation_prompt and not open_turn:
        raise ValueError(
            "add_generation_prompt=True but the conversation already ends with an assistant turn; "
            "drop the trailing response or encode it as a complete conversation."
        )
    if not add_generation_prompt and open_turn:
        raise ValueError(
            f"add_generation_prompt=False but the conversation ends with a {last.get('role')!r} turn, "
            "which the encoder would open a generation prompt after."
        )

# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from twinkle.template.contract.types import (
    ConversationSample, DecodeContext, DecodedResponse, EncodedRow,
    InputOrderCompletionMask, InputOrderLabels, InputOrderLossScale, InputOrderRow,
    KTOSample, MediaBundle, Message, OutputOrderCompletionMask, OutputOrderLabels,
    OutputOrderLossScale, PreferenceSample, PreparedRow, Tool,
)

@dataclass(frozen=True)
class ModelRef:
    model_id_or_dir: str
    revision: Optional[str] = None
    trust_remote_code: bool = True
    processor_kwargs: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class ModelInfo:
    torch_dtype_name: str
    task_type: str
    max_model_len: Optional[int]
    model_type: Optional[str]
    is_multimodal: bool

    @property
    def torch_dtype(self):
        import torch

        return getattr(torch, self.torch_dtype_name)


@dataclass(frozen=True)
class TemplateSpec:
    model: ModelRef
    model_info: ModelInfo
    template_type: str
    agent_template: Optional[str]
    system: Optional[str]
    max_length: Optional[int]
    truncation_strategy: Optional[str]
    max_pixels: Optional[int]
    norm_bbox: Optional[str]
    use_chat_template: Optional[bool]
    padding_side: str
    padding_free: bool
    loss_scale: str
    sequence_parallel_size: int
    is_binary_loss_scale: Optional[bool]
    template_backend: str
    response_prefix: Optional[str]
    enable_thinking: Optional[bool]
    preserve_thinking: Optional[bool]
    add_non_thinking_prefix: bool
    disable_ignore_empty_think: bool
    media_types: Tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TemplateSpec":
        data = dict(value)
        model = data.get("model")
        if isinstance(model, dict):
            processor_kwargs = model.get("processor_kwargs", ())
            model = {**model, "processor_kwargs": tuple(tuple(item) for item in processor_kwargs)}
            data["model"] = ModelRef(**model)
        model_info = data.get("model_info")
        if isinstance(model_info, dict):
            data["model_info"] = ModelInfo(**model_info)
        data["media_types"] = tuple(data["media_types"])
        return cls(**data)

    def replace(self, **changes) -> "TemplateSpec":
        return dataclasses.replace(self, **changes)

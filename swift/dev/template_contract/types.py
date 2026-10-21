# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Optional, Tuple

from typing_extensions import Required, TypedDict

Message = Dict[str, Any]
Tool = Dict[str, Any]


class ConversationSample(TypedDict, total=False):
    messages: Required[List[Message]]
    system: Optional[str]
    tools: List[Tool]
    label: bool
    channel: str
    images: List[Any]
    videos: List[Any]
    audios: List[Any]
    objects: Dict[str, Any]
    chat_template_kwargs: Dict[str, Any]
    extra_kwargs: Dict[str, Any]
    mm_processor_kwargs: Dict[str, Any]


@dataclass(frozen=True)
class PreferenceSample:
    chosen: ConversationSample
    rejected: ConversationSample
    margin: Optional[float] = None


@dataclass(frozen=True)
class KTOSample:
    conversation: ConversationSample
    label: bool


InputOrderLabels = NewType("InputOrderLabels", List[int])
OutputOrderLabels = NewType("OutputOrderLabels", List[int])
InputOrderLossScale = NewType("InputOrderLossScale", List[float])
OutputOrderLossScale = NewType("OutputOrderLossScale", List[float])
InputOrderCompletionMask = NewType("InputOrderCompletionMask", List[int])
OutputOrderCompletionMask = NewType("OutputOrderCompletionMask", List[int])


class InputOrderRow(TypedDict, total=False):
    input_ids: Required[List[int]]
    labels: InputOrderLabels
    loss_scale: InputOrderLossScale
    completion_mask: InputOrderCompletionMask


class EncodedRow(TypedDict, total=False):
    input_ids: Required[List[int]]
    labels: OutputOrderLabels
    loss_scale: OutputOrderLossScale
    completion_mask: OutputOrderCompletionMask


class PreparedRow(EncodedRow, total=False):
    position_ids: Required[Any]
    attention_mask: Any


@dataclass(frozen=True)
class PreferencePair:
    chosen: EncodedRow
    rejected: Optional[EncodedRow]
    margin: Optional[float] = None


@dataclass(frozen=True)
class MediaBundle:
    images: Tuple[Any, ...] = ()
    videos: Tuple[Any, ...] = ()
    audios: Tuple[Any, ...] = ()

    @classmethod
    def from_sources(cls, *sources):
        values = {"images": (), "videos": (), "audios": ()}
        for source in sources:
            if source is None:
                continue
            for key in values:
                if values[key]:
                    continue
                value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
                present = value is not None and (not isinstance(value, (list, tuple)) or bool(value))
                if not present and isinstance(source, dict):
                    value = cls._from_messages(source.get("messages") or (), key)
                    present = bool(value)
                if present:
                    values[key] = tuple(value if isinstance(value, (list, tuple)) else [value])
        return cls(**values)

    @staticmethod
    def _from_messages(messages, key):
        media_type = key[:-1]
        result = []
        for message in messages:
            message_media = message.get(key) or ()
            result.extend(message_media if isinstance(message_media, (list, tuple)) else [message_media])
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") not in {media_type, f"{media_type}_url"}:
                    continue
                value = item.get(item["type"]) or item.get(media_type)
                if isinstance(value, dict):
                    value = value.get("url")
                if value is not None:
                    result.append(value)
        return result

    def __bool__(self) -> bool:
        return bool(self.images or self.videos or self.audios)


@dataclass(frozen=True)
class RolloutState:
    feature: EncodedRow
    engine_input_ids: Tuple[int, ...]
    messages: Tuple[Message, ...]
    tools: Tuple[Tool, ...]
    reasoning_started: bool
    media: MediaBundle
    render_cache: Optional[Any] = None

    def __reduce__(self):
        return (
            type(self),
            (self.feature, self.engine_input_ids, self.messages, self.tools, self.reasoning_started, self.media, None),
        )


@dataclass(frozen=True)
class PromptEncoding:
    state: RolloutState


@dataclass(frozen=True)
class DecodeContext:
    is_finished: bool = True
    first_token: bool = False
    template_inputs: Any = None
    skip_special_tokens: bool = True
    parse_tool_calls: bool = True
    reasoning_started: Optional[bool] = None
    tools: Optional[Tuple[Dict[str, Any], ...]] = None


@dataclass(frozen=True)
class DecodedResponse:
    content: Any
    tool_calls: Tuple[Dict[str, Any], ...] = ()
    errors: Tuple[str, ...] = ()
    reasoning_content: Optional[str] = None
    reasoning_started: Optional[bool] = None


@dataclass(frozen=True)
class CompletionEncoding:
    state: RolloutState
    response: DecodedResponse


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

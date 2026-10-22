# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .types import CompletionEncoding, DecodeContext, DecodedResponse, EncodedRow, InputOrderRow, PromptEncoding


@runtime_checkable
class DatasetTemplate(Protocol):
    def encode(self, sample, **kwargs): ...

    def batch_encode(self, trajectories: List[Any], **kwargs) -> List[Dict[str, Any]]: ...


@runtime_checkable
class PromptEncoder(Protocol):
    def encode(self, sample, **kwargs): ...

    def encode_prompt(self, sample, *, backend: str = "vllm", add_generation_prompt: bool = True) -> PromptEncoding: ...


@runtime_checkable
class Continuation(Protocol):
    def append_completion(
        self,
        prompt,
        tokens: List[int],
        *,
        appended_as: str = "completion",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> CompletionEncoding: ...

    def render_delta(
        self,
        messages_before: List[Dict[str, Any]],
        appended: List[Dict[str, Any]],
        *,
        gen_prompt_before: bool,
        gen_prompt_after: bool,
        tools: Optional[List[Dict[str, Any]]] = None,
        engine: bool = False,
        backend: str = "vllm",
    ) -> List[int]: ...

    def finalize_appended_features(self, input_features: List[InputOrderRow]) -> List[EncodedRow]: ...


@runtime_checkable
class ResponseDecoder(Protocol):
    def decode_response(self, token_ids: List[int], *, context: Optional[DecodeContext] = None) -> DecodedResponse: ...

    def decode_message(self, token_ids: List[int], **kwargs) -> DecodedResponse: ...


@runtime_checkable
class BatchHooks(Protocol):
    def prepare_row(self, row: Dict[str, Any]) -> Dict[str, Any]: ...

    def pop_media(self, row: Dict[str, Any]) -> Dict[str, Any]: ...

    def collate_media(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]: ...

    def finalize_batch(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...


@runtime_checkable
class ModelHook(Protocol):
    def pre_forward_hook(self, model, args, kwargs): ...


@runtime_checkable
class BatchTemplate(BatchHooks, Protocol):
    pass


@runtime_checkable
class SamplerTemplate(PromptEncoder, Continuation, ResponseDecoder, Protocol):
    pass


@runtime_checkable
class TemplateProtocol(DatasetTemplate, SamplerTemplate, ModelHook, Protocol):
    pass

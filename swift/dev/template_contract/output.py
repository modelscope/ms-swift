# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

import json

from .types import DecodedResponse


@runtime_checkable
class ToolParser(Protocol):
    def parse(self, text: str) -> List[Dict[str, Any]]: ...

    def parse_errors(self, text: str) -> List[str]: ...

    def clean(self, text: str) -> str: ...


def split_reasoning_state(
    text: str, *, open_marker: str = "<think>", close_marker: str = "</think>", reasoning_started: bool = False
) -> Tuple[Optional[str], str, bool]:
    state = reasoning_started
    saw_reasoning = reasoning_started
    first_close = None
    offset = 0
    while offset < len(text):
        open_index = text.find(open_marker, offset) if open_marker else -1
        close_index = text.find(close_marker, offset) if close_marker else -1
        if open_index < 0 and close_index < 0:
            break
        if open_index >= 0 and (close_index < 0 or open_index < close_index):
            state = True
            saw_reasoning = True
            offset = open_index + len(open_marker)
        else:
            if first_close is None:
                first_close = close_index
            state = False
            saw_reasoning = True
            offset = close_index + len(close_marker)
    if not saw_reasoning:
        return None, text, state
    if first_close is None:
        head, content = text, ""
    else:
        head = text[:first_close]
        content = text[first_close + len(close_marker) :]
    head = head.lstrip()
    if open_marker and head.startswith(open_marker):
        head = head[len(open_marker) :]
    return head.strip("\n"), content.lstrip("\n"), state


def split_reasoning(
    text: str, *, open_marker: str = "<think>", close_marker: str = "</think>", reasoning_started: bool = False
) -> Tuple[Optional[str], str]:
    reasoning, content, _ = split_reasoning_state(
        text, open_marker=open_marker, close_marker=close_marker, reasoning_started=reasoning_started
    )
    return reasoning, content


def strip_special_tokens(text: str, special_tokens: Iterable[str]) -> str:
    for token in sorted((token for token in special_tokens if token), key=len, reverse=True):
        text = text.replace(token, "")
    return text


def _last_subsequence(sequence, pattern) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    for index in range(len(sequence) - len(pattern), -1, -1):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def _reasoning_is_open(token_ids, tokenizer, open_marker: str, close_marker: str) -> bool:
    ids = list(token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids)

    def encode(marker):
        return list(tokenizer.encode(marker, add_special_tokens=False)) if marker else []

    return _last_subsequence(ids, encode(open_marker)) > _last_subsequence(ids, encode(close_marker))


def _schema_type(schema: Mapping[str, Any]) -> Optional[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return next((value for value in schema_type if value != "null"), None)
    if isinstance(schema_type, str):
        return schema_type
    for candidate in schema.get("anyOf") or schema.get("oneOf") or ():
        candidate_type = _schema_type(candidate)
        if candidate_type and candidate_type != "null":
            return candidate_type
    return None


def _coerce_schema_value(value: Any, schema: Mapping[str, Any]) -> Any:
    schema_type = _schema_type(schema)
    try:
        if schema_type == "integer" and isinstance(value, str):
            return int(value)
        if schema_type == "number" and isinstance(value, str):
            return float(value)
        if schema_type == "boolean" and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1"}:
                return True
            if lowered in {"false", "0"}:
                return False
        if schema_type in {"array", "object"} and isinstance(value, str):
            value = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value
    if schema_type == "object" and isinstance(value, dict):
        properties = schema.get("properties") or {}
        return {
            key: _coerce_schema_value(item, properties[key]) if key in properties else item
            for key, item in value.items()
        }
    if schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items") or {}
        return [_coerce_schema_value(item, item_schema) for item in value]
    return value


def coerce_tool_call_arguments(
    calls: Iterable[Dict[str, Any]], tools: Optional[Iterable[Dict[str, Any]]]
) -> Tuple[Dict[str, Any], ...]:
    schemas = {}
    for tool in tools or ():
        function = tool.get("function", tool)
        if function.get("name"):
            schemas[function["name"]] = function.get("parameters") or {}
    result = []
    for call in calls:
        function = dict(call.get("function") or {})
        name = function.get("name") or call.get("name")
        arguments = function.get("arguments", call.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                pass
        if name in schemas:
            arguments = _coerce_schema_value(arguments, schemas[name])
        result.append({**call, "function": {**function, "name": name, "arguments": arguments}})
    return tuple(result)


def decode_output(
    raw_text: str,
    *,
    thinking_marks: Optional[Tuple[str, str]] = None,
    reasoning_started: bool = False,
    tool_parser: Optional[ToolParser] = None,
    tools: Optional[Tuple[Dict[str, Any], ...]] = None,
    special_tokens: Tuple[str, ...] = (),
    require_parser_for_tools: bool = False,
    structured_calls: Optional[Iterable[Dict[str, Any]]] = None,
) -> DecodedResponse:
    if thinking_marks is None:
        reasoning, content = None, raw_text
    else:
        open_marker, close_marker = (marker.strip() for marker in thinking_marks)
        reasoning, content, reasoning_started = split_reasoning_state(
            raw_text, open_marker=open_marker, close_marker=close_marker, reasoning_started=reasoning_started
        )

    raw_calls = ()
    if structured_calls is not None:
        raw_calls = structured_calls
        errors = ()
    elif tool_parser is not None:
        raw_calls = tool_parser.parse(content)
        errors = tuple(tool_parser.parse_errors(content))
        content = tool_parser.clean(content)
    else:
        if require_parser_for_tools and tools:
            raise RuntimeError("tools were provided but no tool-call parser is configured.")
        errors = ()
    calls = coerce_tool_call_arguments(raw_calls, tools)

    if special_tokens:
        content = strip_special_tokens(content, special_tokens)
        if reasoning is not None:
            reasoning = strip_special_tokens(reasoning, special_tokens)
    return DecodedResponse(
        content=content,
        tool_calls=calls,
        errors=errors,
        reasoning_content=reasoning,
        reasoning_started=reasoning_started if thinking_marks is not None else None,
    )

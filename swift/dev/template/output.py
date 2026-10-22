# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import ast
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json

from swift.dev.template_contract import DecodeContext, DecodedResponse
from swift.dev.template_contract.output import ToolParser, decode_output


def _arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    for loads in (json.loads, ast.literal_eval):
        try:
            return loads(text)
        except (ValueError, SyntaxError, TypeError):
            continue
    return text


class AgentToolParser:
    def __init__(self, agent):
        self.agent = agent

    def parse(self, text: str) -> List[Dict[str, Any]]:
        return self._scan(text)[0]

    def parse_errors(self, text: str) -> List[str]:
        return self._scan(text)[1]

    def clean(self, text: str) -> str:
        text = text or ""
        calls = self.parse(text)
        if not calls:
            return text.rstrip()
        index = self._call_index(text, calls)
        return text[:index].rstrip() if index is not None else text.rstrip()

    def _call_index(self, text: str, calls: List[Dict[str, Any]]) -> Optional[int]:
        from .agent.base import ReactCompatMixin

        index = self.agent.find_tool_call_start(text, calls)
        if index is not None:
            return index
        marker = ReactCompatMixin.keyword.action
        return text.find(marker) if marker in text else None

    def _scan(self, text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        text = text or ""
        try:
            functions = self.agent.get_toolcall(text)
        except Exception as e:  # noqa: BLE001
            return [], [f"{type(e).__name__}: {e}"]
        calls = [
            {
                "type": "function",
                "function": {"name": function.name.strip(), "arguments": _arguments(function.arguments)},
            }
            for function in functions
        ]
        start = self.agent.tool_call_start
        if not calls and start and start in text:
            return [], [f"the response opens a tool call ({start!r}) but no call could be parsed"]
        return calls, []


class OutputCodec:
    def __init__(self, parser: Optional[AgentToolParser] = None):
        self.parser = parser

    def decode(
        self,
        template,
        token_ids: List[int],
        context: DecodeContext,
        structured_calls: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> DecodedResponse:
        kwargs = {
            "is_finished": context.is_finished,
            "first_token": context.first_token,
            "skip_special_tokens": False,
        }
        if context.template_inputs is not None:
            kwargs["template_inputs"] = context.template_inputs
        content = template.decode_generate_ids(token_ids, **kwargs)
        if not isinstance(content, str):
            return DecodedResponse(content=content)
        reasoning_started = context.reasoning_started
        if reasoning_started is None:
            reasoning_started = bool(template._get_enable_thinking(context.template_inputs))
        meta = template.template_meta
        has_thinking = meta.is_thinking or meta.thinking_prefix or meta.non_thinking_prefix
        return decode_output(
            content,
            thinking_marks=template._thinking_marks() if has_thinking else None,
            reasoning_started=reasoning_started,
            tool_parser=self.parser if context.parse_tool_calls and structured_calls is None else None,
            tools=context.tools,
            special_tokens=tuple(getattr(template.tokenizer, "all_special_tokens", ()))
            if context.skip_special_tokens
            else (),
            require_parser_for_tools=context.parse_tool_calls,
            structured_calls=structured_calls,
        )

    def parse_tool_call(self, decoded: str) -> List[Dict[str, Any]]:
        return self.parser.parse(decoded) if self.parser else []

    def clean_tool_call(self, decoded: str) -> str:
        return self.parser.clean(decoded) if self.parser else (decoded or "").rstrip()

    def tool_call_errors(self, decoded: str) -> List[str]:
        return self.parser.parse_errors(decoded) if self.parser else []

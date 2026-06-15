# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Any, List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate

# Special token used as a namespace prefix for every XML tag in MiniMax-M3
# tool_call payloads.
NS_TOKEN = ']<]minimax[>['
TOOLCALL_BEGIN_TOKEN = NS_TOKEN + '<tool_call>'
TOOLCALL_END_TOKEN = NS_TOKEN + '</tool_call>'


def _to_xml(val: Any, ns: str = NS_TOKEN) -> str:
    """Recursive XML renderer mirroring the ``to_xml`` macro in MiniMax-M3's
    ``chat_template.jinja``.

    ``None`` values are intentionally omitted (consistent with the upstream
    convention that drops ``None`` parameters rather than emitting a literal
    ``null`` string).
    """
    if val is None:
        return ''
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            if v is None:
                continue
            parts.append(f'{ns}<{k}>{_to_xml(v, ns)}{ns}</{k}>')
        return ''.join(parts)
    if isinstance(val, (list, tuple)):
        parts = []
        for item in val:
            parts.append(f'{ns}<item>{_to_xml(item, ns)}{ns}</item>')
        return ''.join(parts)
    if isinstance(val, bool):
        return json.dumps(val)
    return str(val)


_NS = re.escape(NS_TOKEN)
_TC_BEGIN = re.escape(TOOLCALL_BEGIN_TOKEN)
_TC_END = re.escape(TOOLCALL_END_TOKEN)
# Match any opening tag like ]<]minimax[>[<key> (excluding closing/invoke/tool_call)
_INVOKE_RE = re.compile(rf'{_NS}<invoke\s+name="([^"]+)">(.*?){_NS}</invoke>', re.DOTALL)
_TOOLCALL_RE = re.compile(rf'{_TC_BEGIN}(.*?){_TC_END}', re.DOTALL)


def _parse_xml_value(content: str) -> Any:
    """Parse an XML fragment produced by ``to_xml`` back into a Python value.

    The expected fragments use ``NS_TOKEN`` as a tag prefix. The function
    handles nested ``<item>`` lists, dict-like ``<key>...</key>`` structures
    and falls back to a stripped string for primitive payloads.
    """
    content = content.strip()
    if not content:
        return ''

    # Try list of items first (heuristic: starts with `<item>`).
    if content.startswith(f'{NS_TOKEN}<item>'):
        items = []
        for inner in _iter_tagged(content, 'item'):
            items.append(_parse_xml_value(inner))
        return items

    # Try mapping (heuristic: starts with a NS_TOKEN<tag>).
    if content.startswith(NS_TOKEN + '<'):
        result: dict = {}
        for key, inner in _iter_keyed(content):
            result[key] = _parse_xml_value(inner)
        if result:
            return result

    # Primitive fallback. Try JSON (booleans / numbers) before raw text.
    try:
        return json.loads(content)
    except Exception:
        return content


def _iter_tagged(content: str, tag: str):
    pattern = re.compile(rf'{_NS}<{re.escape(tag)}>(.*?){_NS}</{re.escape(tag)}>', re.DOTALL)
    for m in pattern.finditer(content):
        yield m.group(1)


def _iter_keyed(content: str):
    """Iterate ``(tag_name, inner_content)`` for top-level NS-prefixed tags."""
    cursor = 0
    n = len(content)
    open_pat = re.compile(rf'{_NS}<([^/!?\s>]+)>')
    while cursor < n:
        m = open_pat.search(content, cursor)
        if not m:
            return
        name = m.group(1)
        end_marker = f'{NS_TOKEN}</{name}>'
        # Match nested same-name tags by counting depth.
        depth = 1
        scan = m.end()
        open_marker = f'{NS_TOKEN}<{name}>'
        while depth > 0 and scan < n:
            next_open = content.find(open_marker, scan)
            next_close = content.find(end_marker, scan)
            if next_close == -1:
                return
            if next_open != -1 and next_open < next_close:
                depth += 1
                scan = next_open + len(open_marker)
            else:
                depth -= 1
                scan = next_close + len(end_marker)
        inner = content[m.end():scan - len(end_marker)]
        yield name, inner
        cursor = scan


class MinimaxM3AgentTemplate(BaseAgentTemplate):
    """Agent template for MiniMax-M3 series multimodal models.

    Tool calls follow this XML-with-namespace format:

        ]<]minimax[>[<tool_call>
        ]<]minimax[>[<invoke name="tool-name">
        ]<]minimax[>[<param-1>value-1]<]minimax[>[</param-1>
        ]<]minimax[>[<param-2>]<]minimax[>[<item>...]<]minimax[>[</item>]<]minimax[>[</param-2>
        ]<]minimax[>[</invoke>
        ]<]minimax[>[</tool_call>

    Tool responses are wrapped in ``<response>...</response>`` inside a
    ``]~b]tool`` slot.
    """

    def get_toolcall(self, response: str) -> List[Function]:
        functions: List[Function] = []
        for tc_block in _TOOLCALL_RE.findall(response):
            for tool_name, params_block in _INVOKE_RE.findall(tc_block):
                arguments = {}
                for key, inner in _iter_keyed(params_block):
                    arguments[key] = _parse_xml_value(inner)
                functions.append(Function(name=tool_name, arguments=arguments))

        if not functions:
            return super().get_toolcall(response)
        return functions

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)

        if hasattr(self, 'template_meta'):
            prompt = self.template_meta.prompt.copy()
            chat_sep = self.template_meta.chat_sep
            for i in range(len(prompt)):
                if isinstance(prompt[i], str):
                    prompt[i] = prompt[i].replace('user', 'tool')
        else:
            prompt = [']~b]tool\n{{QUERY}}[e~[\n]~b]ai\n']
            chat_sep = ['[e~[\n']

        res = chat_sep.copy() if chat_sep else []
        tool_responses = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            tool_responses.append(f'<response>{tool_content}</response>')
        total_tool = '\n'.join(tool_responses)

        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool)
            res.append(context)
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_schemas = []
        for tool in tools:
            tool = self.unwrap_tool(tool)
            tool_schemas.append(json.dumps(tool, ensure_ascii=False))

        system = system or ''
        tools_xml = '\n'.join(f'<tool>{schema}</tool>' for schema in tool_schemas)
        # Mirror the example block produced by chat_template.jinja so the
        # in-context format hint matches inference time exactly.
        # Note: jinja emits 'Example:\n' then '\n' before the tool_call_begin
        # token, which renders as two consecutive newlines.
        example = (f'\n\n{TOOLCALL_BEGIN_TOKEN}\n'
                   f'{NS_TOKEN}<invoke name="tool-name-1">'
                   f'{NS_TOKEN}<param-1>value-1{NS_TOKEN}</param-1>'
                   f'{NS_TOKEN}<param-2>'
                   f'{NS_TOKEN}<item>'
                   f'{NS_TOKEN}<key-a>val-a{NS_TOKEN}</key-a>'
                   f'{NS_TOKEN}<key-b>val-b{NS_TOKEN}</key-b>'
                   f'{NS_TOKEN}</item>'
                   f'{NS_TOKEN}</param-2>'
                   f'{NS_TOKEN}</invoke>\n'
                   f'{NS_TOKEN}<invoke name="tool-name-2">'
                   f'{NS_TOKEN}<param-1>value-1{NS_TOKEN}</param-1>'
                   f'{NS_TOKEN}</invoke>\n'
                   f'{TOOLCALL_END_TOKEN}')

        return (f'{system}\n\n# Tools\n'
                'You may call one or more tools to assist with the user query.\n'
                'Here are the tools available in JSONSchema format:\n'
                f'\n<tools>\n{tools_xml}\n</tools>\n\n'
                f'To call tools, wrap all invocations in a single {TOOLCALL_BEGIN_TOKEN}{TOOLCALL_END_TOKEN} '
                'block. Parameter values containing nested objects or arrays are recursively expanded into '
                f'XML elements. Example:{example}')

    def _format_tool_calls(self, tool_call_messages) -> str:
        invocations = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = tool_call['arguments'] or {}

            param_parts = [f'{NS_TOKEN}<invoke name="{name}">']
            for k, v in arguments.items():
                if v is None:
                    continue
                param_parts.append(f'{NS_TOKEN}<{k}>{_to_xml(v, NS_TOKEN)}{NS_TOKEN}</{k}>')
            param_parts.append(f'{NS_TOKEN}</invoke>')
            invocations.append(''.join(param_parts))

        if not invocations:
            return ''
        return f'{TOOLCALL_BEGIN_TOKEN}\n' + '\n'.join(invocations) + f'\n{TOOLCALL_END_TOKEN}'

# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.infer_engine import Function
from swift.template import Prompt
from .base import BaseAgentTemplate

QUOTE = '<|"|>'
_STANDARD_KEYS = {'description', 'type', 'properties', 'required', 'nullable'}


class Gemma4AgentTemplate(BaseAgentTemplate):
    """Agent template for Google Gemma-4 models.

    Reference: chat_template.jinja shipped with google/gemma-4-12B-it.
    Tool definitions are wrapped in `<|tool>...<tool|>` and rendered with the
    custom DSL described by the official chat template.
    Tool calls follow `<|tool_call>call:NAME{key:value,...}<tool_call|>` and
    tool responses follow `<|tool_response>response:NAME{...}<tool_response|>`.
    """

    @classmethod
    def _format_argument(cls, value: Any, escape_keys: bool = True) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, str):
            return f'{QUOTE}{value}{QUOTE}'
        if value is None:
            return 'null'
        if isinstance(value, dict):
            items = []
            for k in sorted(value.keys()):
                v = value[k]
                key_str = f'{QUOTE}{k}{QUOTE}' if escape_keys else str(k)
                items.append(f'{key_str}:{cls._format_argument(v, escape_keys=escape_keys)}')
            return '{' + ','.join(items) + '}'
        if isinstance(value, (list, tuple)):
            return '[' + ','.join(cls._format_argument(item, escape_keys=escape_keys) for item in value) + ']'
        return str(value)

    @classmethod
    def _format_parameters(cls,
                           properties: Dict[str, Any],
                           required: Optional[List[str]] = None,
                           filter_keys: bool = False) -> str:
        parts = []
        for key in sorted(properties.keys()):
            value = properties[key]
            if filter_keys and key in _STANDARD_KEYS:
                continue
            if not isinstance(value, dict):
                continue
            inner: List[str] = []
            type_upper = (value.get('type') or '').upper() if isinstance(value.get('type'), str) else ''
            if value.get('description'):
                inner.append(f'description:{QUOTE}{value["description"]}{QUOTE}')
            if type_upper == 'STRING':
                if value.get('enum'):
                    inner.append(f'enum:{cls._format_argument(value["enum"])}')
            elif type_upper == 'ARRAY':
                items_value = value.get('items')
                if isinstance(items_value, dict) and items_value:
                    items_inner: List[str] = []
                    items_required = items_value.get('required', [])
                    for item_key in sorted(items_value.keys()):
                        item_value = items_value[item_key]
                        if item_value is None:
                            continue
                        if item_key == 'properties' and isinstance(item_value, dict):
                            items_inner.append(f'properties:{{{cls._format_parameters(item_value, items_required)}}}')
                        elif item_key == 'required':
                            req_str = ','.join(f'{QUOTE}{r}{QUOTE}' for r in item_value)
                            items_inner.append(f'required:[{req_str}]')
                        elif item_key == 'type':
                            if isinstance(item_value, str):
                                items_inner.append(f'type:{cls._format_argument(item_value.upper())}')
                            else:
                                items_inner.append(f'type:{cls._format_argument([str(t).upper() for t in item_value])}')
                        else:
                            items_inner.append(f'{item_key}:{cls._format_argument(item_value)}')
                    inner.append('items:{' + ','.join(items_inner) + '}')
            if value.get('nullable'):
                inner.append('nullable:true')
            if type_upper == 'OBJECT':
                inner_required = value.get('required', [])
                if isinstance(value.get('properties'), dict):
                    inner.append(f'properties:{{{cls._format_parameters(value["properties"], inner_required)}}}')
                else:
                    inner.append(f'properties:{{{cls._format_parameters(value, inner_required, filter_keys=True)}}}')
                if value.get('required'):
                    req_str = ','.join(f'{QUOTE}{r}{QUOTE}' for r in value['required'])
                    inner.append(f'required:[{req_str}]')
            inner.append(f'type:{QUOTE}{type_upper}{QUOTE}')
            parts.append(f'{key}:{{{",".join(inner)}}}')
        return ','.join(parts)

    @classmethod
    def _format_function_declaration(cls, tool: Dict[str, Any]) -> str:
        function = tool['function']
        name = function.get('name', '')
        description = function.get('description', '') or ''
        result = f'declaration:{name}{{description:{QUOTE}{description}{QUOTE}'
        params = function.get('parameters')
        if params:
            param_parts: List[str] = []
            properties = params.get('properties')
            if properties:
                param_parts.append(f'properties:{{{cls._format_parameters(properties, params.get("required", []))}}}')
            if params.get('required'):
                req_str = ','.join(f'{QUOTE}{r}{QUOTE}' for r in params['required'])
                param_parts.append(f'required:[{req_str}]')
            ptype = params.get('type')
            if isinstance(ptype, str) and ptype:
                param_parts.append(f'type:{QUOTE}{ptype.upper()}{QUOTE}')
            if param_parts:
                result += ',parameters:{' + ','.join(param_parts) + '}'
        result += '}'
        return result

    def _format_tools(self,
                      tools: List[Union[str, dict]],
                      system: Optional[str] = None,
                      user_message: Optional[dict] = None) -> str:
        tool_blocks: List[str] = []
        for tool in tools:
            tool = self.wrap_tool(tool)
            tool_blocks.append(f'<|tool>{self._format_function_declaration(tool)}<tool|>')
        system_text = (system or '').strip()
        return system_text + ''.join(tool_blocks)

    def _format_tool_calls(self, tool_call_messages) -> str:
        invocations: List[str] = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = tool_call['arguments']
            if isinstance(arguments, str):
                arguments = self._parse_json(arguments) or {}
            if isinstance(arguments, dict):
                args_str = ','.join(f'{k}:{self._format_argument(arguments[k], escape_keys=False)}'
                                    for k in sorted(arguments.keys()))
            else:
                args_str = ''
            invocations.append(f'<|tool_call>call:{name}{{{args_str}}}<tool_call|>')
        return ''.join(invocations)

    def _get_tool_responses(self, tool_messages) -> str:
        parts: List[str] = []
        for tool_message in tool_messages:
            tool_name = tool_message.get('name') or 'unknown'
            tool_content = tool_message.get('content')
            if isinstance(tool_content, dict):
                inner = ','.join(f'{k}:{self._format_argument(tool_content[k], escape_keys=False)}'
                                 for k in sorted(tool_content.keys()))
                parts.append(f'<|tool_response>response:{tool_name}{{{inner}}}<tool_response|>')
            else:
                # Match jinja: treat string/other content as a single `value:` field.
                value = '' if tool_content is None else tool_content
                parts.append(f'<|tool_response>response:{tool_name}'
                             f'{{value:{self._format_argument(value, escape_keys=False)}}}<tool_response|>')
        return ''.join(parts)

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        # If the model hallucinated a trailing `<|tool_response>` opener (e.g. when stop
        # tokens were not configured), strip it so the rendered turn does not contain
        # `<|tool_response><|tool_response>response:...`.
        if assistant_content.endswith('<|tool_response>'):
            assistant_content = assistant_content[:-len('<|tool_response>')]
        # In gemma4, tool_call/tool_response/follow-up assistant text all live in the
        # same `<|turn>model ... <turn|>` block, so we do not open a new model turn here.
        res: 'Prompt' = [self._get_tool_responses(tool_messages)]
        return assistant_content, res

    @classmethod
    def _gemma_to_json(cls, s: str) -> str:
        # `<|"|>` -> `"`; bare keys preceded by `{` or `,` get JSON-quoted.
        s = s.replace(QUOTE, '"')
        s = re.sub(r'(?<=[\{,])([A-Za-z_][\w\-]*)(?=:)', r'"\1"', s)
        return s

    @classmethod
    def _parse_arguments(cls, args_body: str) -> Dict[str, Any]:
        json_str = cls._gemma_to_json('{' + args_body + '}')
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {}

    def get_toolcall(self, response: str) -> List[Function]:
        pattern = re.compile(r'<\|tool_call>call:([^\{]+)\{(.*?)\}<tool_call\|>', re.DOTALL)
        functions: List[Function] = []
        for match in pattern.finditer(response):
            name = match.group(1).strip()
            arguments = self._parse_arguments(match.group(2))
            functions.append(Function(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
        if not functions:
            return super().get_toolcall(response)
        return functions

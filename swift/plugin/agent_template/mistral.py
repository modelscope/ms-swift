import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt


class MistralAgentTemplate(BaseAgentTemplate):

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer import Function
        res_list = re.findall(r'\[TOOL_CALLS\]\[(.*?)\]</s>', response, re.DOTALL)
        if not res_list:
            return []
        res_list = res_list[0].strip().split('\n')
        functions = []
        for res_str in res_list:
            parsed_res = self._parse_json(res_str)
            if isinstance(parsed_res, dict):
                parsed_res = [parsed_res]  # Handle single tool call
            if isinstance(parsed_res, list):
                for tool_call in parsed_res:
                    if isinstance(tool_call, dict) and 'name' in tool_call and 'arguments' in tool_call:
                        functions.append(Function(name=tool_call['name'], arguments=tool_call['arguments']))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        if not hasattr(self, 'template_meta'):
            raise ValueError('MistralAgentTemplate requires template_meta to be registered')
        prompt = self.template_meta.prompt
        chat_sep = self.template_meta.chat_sep

        res = chat_sep.copy()
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            # append `[TOOL_RESULTS]{"content": {{ .Content }}}[/TOOL_RESULTS]` to res_tool
            res_tool.append(f'[TOOL_RESULTS]{json.dumps({"content": tool_content}, ensure_ascii=False)}[/TOOL_RESULTS]')
        total_tool = '\n'.join(res_tool)
        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool)
            res.append(context)
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: Optional[str] = None, user_message=None) -> str:
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        system = system or ''
        return f"""{system}[AVAILABLE_TOOLS]{' '.join(tool_descs)}[/AVAILABLE_TOOLS]"""

    def _format_tool_calls(self, tool_call_messages):
        tool_calls = []
        for message in tool_call_messages:
            # needs `{'name': name, 'arguments': arguments}`, which self._parse_tool_call
            # satisfies
            tool_call = self._parse_tool_call(message['content'])
            tool_calls.append(json.dumps(tool_call, ensure_ascii=False))
        return f'[TOOL_CALLS][\n{chr(10).join(tool_calls)}\n]</s>'  # check if need `</s>` at end

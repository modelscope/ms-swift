# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import unittest

from swift.agent_template import agent_template_map

# Kimi K2/K2.5 tool-call protocol (chat_template.jinja + docs/tool_call_guidance.md in the model repo):
#   <|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{idx}<|tool_call_argument_begin|>{args}
#   <|tool_call_end|><|tool_calls_section_end|>
# The segment before <|tool_call_argument_begin|> is the tool call *id* `functions.{name}:{idx}`, and the
# function name is recovered with `tool_call_id.split('.')[1].split(':')[0]`. The tool result is rendered
# as `## Return of {tool_call_id}`.
CALL_ID = 'functions.get_weather:0'
MODEL_OUTPUT = ('<|tool_calls_section_begin|>'
                f'<|tool_call_begin|>{CALL_ID}<|tool_call_argument_begin|>{{"city": "Beijing"}}<|tool_call_end|>'
                '<|tool_calls_section_end|>')


class TestKimiK25ToolCallId(unittest.TestCase):

    def setUp(self):
        self.agent = agent_template_map['kimi_k25']()
        self.tool_call_message = {
            'role': 'tool_call',
            'content': json.dumps({
                'name': 'get_weather',
                'arguments': {
                    'city': 'Beijing'
                }
            })
        }

    def test_get_toolcall_returns_function_name_not_call_id(self):
        functions = self.agent.get_toolcall(MODEL_OUTPUT)
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0].name, 'get_weather')
        self.assertEqual(json.loads(functions[0].arguments), {'city': 'Beijing'})

    def test_format_tool_calls_renders_kimi_call_id(self):
        rendered = self.agent._format_tool_calls([self.tool_call_message])
        self.assertEqual(rendered, MODEL_OUTPUT)

    def test_tool_response_carries_call_id(self):
        assistant = self.agent._format_tool_calls([self.tool_call_message])
        _, responses = self.agent._format_tool_responses(assistant, [{'role': 'tool', 'content': 'sunny'}])
        self.assertIn(f'## Return of {CALL_ID}\nsunny', ''.join(responses))

    def test_parallel_tool_calls(self):
        tool_call_messages = [
            self.tool_call_message, {
                'role': 'tool_call',
                'content': json.dumps({
                    'name': 'get_time',
                    'arguments': {
                        'city': 'Beijing'
                    }
                })
            }
        ]
        assistant = self.agent._format_tool_calls(tool_call_messages)
        self.assertIn('<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>', assistant)
        self.assertEqual([f.name for f in self.agent.get_toolcall(assistant)], ['get_weather', 'get_time'])
        # assistant content may be a list after consecutive assistant messages are merged
        _, responses = self.agent._format_tool_responses(['Let me check.', assistant], [{
            'role': 'tool',
            'content': 'sunny'
        }, {
            'role': 'tool',
            'content': '12:00'
        }])
        responses = ''.join(responses)
        self.assertIn('## Return of functions.get_weather:0\nsunny', responses)
        self.assertIn('## Return of functions.get_time:1\n12:00', responses)

    def test_get_toolcall_accepts_bare_function_name(self):
        response = '<|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"city": "Beijing"}<|tool_call_end|>'
        self.assertEqual(self.agent.get_toolcall(response)[0].name, 'get_weather')


if __name__ == '__main__':
    unittest.main()

# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import unittest

from swift.agent_template.gemma4 import Gemma4AgentTemplate


class TestGemma4ToolCall(unittest.TestCase):

    def setUp(self):
        self.agent = Gemma4AgentTemplate()

    def _round_trip(self, arguments):
        content = json.dumps({'name': 'run', 'arguments': arguments})
        response = self.agent._format_tool_calls([{'role': 'tool_call', 'content': content}])
        functions = self.agent.get_toolcall(response)
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0].name, 'run')
        return json.loads(functions[0].arguments)

    def test_string_arguments_are_not_escaped(self):
        # Gemma-4 wraps string values in `<|"|>` without escaping their content.
        for arguments in (
            {
                'path': 'a.py',
                'content': 'print("hi")\n'
            },
            {
                'cmd': 'ls\ncd /tmp'
            },
            {
                'text': 'a\tb\\c'
            },
            {
                'q': 'a,b:c'
            },
            {
                'q': 'x{y}'
            },
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(self._round_trip(arguments), arguments)

    def test_nested_arguments(self):
        arguments = {'opts': {'name': 'say "hi"', 'tags': ['a,b:c', 1, True, None]}, 'n': 3, 'f': 1.5}
        self.assertEqual(self._round_trip(arguments), arguments)

    def test_parse_model_output(self):
        response = '<|tool_call>call:write{content:<|"|>x = "1"\ny = 2<|"|>,path:<|"|>a.py<|"|>}<tool_call|>'
        functions = self.agent.get_toolcall(response)
        self.assertEqual([f.name for f in functions], ['write'])
        self.assertEqual(json.loads(functions[0].arguments), {'content': 'x = "1"\ny = 2', 'path': 'a.py'})


if __name__ == '__main__':
    unittest.main()

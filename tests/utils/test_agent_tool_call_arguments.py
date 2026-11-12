# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import unittest

from swift.agent_template import agent_template_map


class TestAgentToolCallArguments(unittest.TestCase):

    def test_tool_call_arguments_are_json(self):
        # The arguments must be rendered as JSON (as the tool prompts ask for), not as a Python repr,
        # so that the tool call parsed back from the model output is valid JSON.
        arguments = {'city': '北京', 'exact': True, 'limit': None}
        tool_call_messages = [{'role': 'tool_call', 'content': json.dumps({'name': 'weather', 'arguments': arguments})}]
        for name in ['react_en', 'react_zh', 'qwen_en', 'qwen_zh', 'toolbench', 'chatglm4', 'glm4']:
            with self.subTest(agent_template=name):
                agent_template = agent_template_map[name]()
                content = agent_template._format_tool_calls(tool_call_messages)
                self.assertIn(json.dumps(arguments, ensure_ascii=False), content)
                functions = agent_template.get_toolcall(content)
                self.assertEqual(len(functions), 1)
                self.assertEqual(functions[0].name, 'weather')
                self.assertEqual(json.loads(functions[0].arguments), arguments)


if __name__ == '__main__':
    unittest.main()

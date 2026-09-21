# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import unittest

from swift.agent_template import agent_template_map


class TestGLM4_5ToolCallArguments(unittest.TestCase):
    # The GLM-4.5/4.7/5.1 chat templates render each argument with
    # `{{ v | tojson(ensure_ascii=False) if v is not string else v }}`.

    def test_non_string_arguments_are_rendered_as_json(self):
        arguments = {'city': '北京', 'days': 3, 'metric': True, 'extra': None, 'filters': {'tags': ['a', 'b']}}
        message = {'role': 'tool_call', 'content': json.dumps({'name': 'get_weather', 'arguments': arguments})}
        for name in ['glm4_5', 'glm4_7', 'glm5_1']:
            with self.subTest(agent_template=name):
                rendered = agent_template_map[name]()._format_tool_calls([message])
                self.assertIn('<arg_value>北京</arg_value>', rendered)
                self.assertIn('<arg_value>3</arg_value>', rendered)
                self.assertIn('<arg_value>true</arg_value>', rendered)
                self.assertIn('<arg_value>null</arg_value>', rendered)
                self.assertIn('<arg_value>{"tags": ["a", "b"]}</arg_value>', rendered)


if __name__ == '__main__':
    unittest.main()

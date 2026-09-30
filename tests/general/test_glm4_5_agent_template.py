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


class TestGLM4_5ToolCallPrefix(unittest.TestCase):
    # The GLM-4.5/4.6 chat_template renders `{{ '\n<tool_call>' + tc.name }}` for
    # every tool call: a '\n' separates the assistant's non-empty content from
    # <tool_call>. The GLM-5 style templates (glm4_7/glm5_1) have no separator.

    def test_non_empty_content_gets_newline_separator(self):
        agent_template = agent_template_map['glm4_5']()
        prefix = agent_template._add_tool_call_prefix('<tool_call>get_current_weather</tool_call>', {
            'role': 'assistant',
            'content': 'Let me check.'
        })
        self.assertEqual(prefix, '\n<tool_call>get_current_weather</tool_call>')

    def test_empty_content_has_no_separator(self):
        agent_template = agent_template_map['glm4_5']()
        for content in ['', '   ', None, ['Let me check.']]:
            with self.subTest(content=content):
                prefix = agent_template._add_tool_call_prefix('<tool_call>call</tool_call>', {
                    'role': 'assistant',
                    'content': content,
                })
                self.assertEqual(prefix, '<tool_call>call</tool_call>')

    def test_non_assistant_pre_message_has_no_separator(self):
        agent_template = agent_template_map['glm4_5']()
        for pre_message in [None, {'role': 'user', 'content': 'query'}]:
            with self.subTest(pre_message=pre_message):
                prefix = agent_template._add_tool_call_prefix('<tool_call>call</tool_call>', pre_message)
                self.assertEqual(prefix, '<tool_call>call</tool_call>')

    def test_glm5_style_templates_have_no_separator(self):
        for name in ['glm4_7', 'glm5_1']:
            with self.subTest(agent_template=name):
                agent_template = agent_template_map[name]()
                prefix = agent_template._add_tool_call_prefix('<tool_call>call</tool_call>', {
                    'role': 'assistant',
                    'content': 'Let me check.'
                })
                self.assertEqual(prefix, '<tool_call>call</tool_call>')


if __name__ == '__main__':
    unittest.main()

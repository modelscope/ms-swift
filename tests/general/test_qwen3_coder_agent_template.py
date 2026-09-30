# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest

from swift.agent_template import agent_template_map


class TestQwen3CoderToolCallPrefix(unittest.TestCase):
    # The Qwen3-Coder chat_template renders `'\n' + content|trim + '\n'` before
    # `'\n<tool_call>'`: a '\n\n' separator sits between the assistant's non-empty
    # content and <tool_call>. Qwen3.5/3.6 override this in their own class.

    def test_non_empty_content_gets_separator(self):
        agent_template = agent_template_map['qwen3_coder']()
        prefix = agent_template._add_tool_call_prefix(
            '<tool_call>\n<function=get_current_weather>\n</function>\n</tool_call>',
            {'role': 'assistant', 'content': 'Let me check.'})
        self.assertEqual(prefix, '\n\n<tool_call>\n<function=get_current_weather>\n</function>\n</tool_call>')

    def test_empty_content_has_no_separator(self):
        agent_template = agent_template_map['qwen3_coder']()
        for content in ['', '   ', None, ['Let me check.']]:
            with self.subTest(content=content):
                prefix = agent_template._add_tool_call_prefix('<tool_call>call</tool_call>', {
                    'role': 'assistant',
                    'content': content,
                })
                self.assertEqual(prefix, '<tool_call>call</tool_call>')

    def test_non_assistant_pre_message_has_no_separator(self):
        agent_template = agent_template_map['qwen3_coder']()
        for pre_message in [None, {'role': 'user', 'content': 'query'}]:
            with self.subTest(pre_message=pre_message):
                prefix = agent_template._add_tool_call_prefix('<tool_call>call</tool_call>', pre_message)
                self.assertEqual(prefix, '<tool_call>call</tool_call>')


if __name__ == '__main__':
    unittest.main()

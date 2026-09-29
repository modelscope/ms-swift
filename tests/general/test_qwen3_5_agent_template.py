# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for #10255.

The Qwen3.5/3.6 swift backend was dropping the assistant's ``<think>...</think>``
reasoning whenever the same assistant turn also carried ``tool_calls`` (and, in
the history-thinking path, whenever a tool message sat between the user prompt
and the assistant reasoning). The official ``chat_template.jinja`` (used by
vLLM / transformers at inference) keeps both, which is the root cause of the
train/infer mismatch on Qwen3.5/3.6 agent data (the same upstream issue as
#9234 — ``</think><|im_end|>`` / empty ``<​tool_call>`` after SFT).

These tests exercise the agent-template and template-base hooks directly,
without loading a model or downloading tokenizer files, so they run on CPU
only.
"""
import json
import unittest

from swift.agent_template import agent_template_map
from swift.template.base import Template
from swift.template.utils import get_last_user_round


class TestQwen3_5AddToolCallPrefix(unittest.TestCase):
    """The Qwen3.5/3.6 jinja keeps the preceding assistant ``content`` (including
    ``<think>...</think>`` reasoning) before ``<​tool_call>`` and only inserts
    ``\\n\\n`` between them when the effective (post-think) content is
    non-empty. The previous swift implementation used ``pre_message['content']``
    only to decide on the separator and dropped the reasoning entirely — see
    #10255.
    """

    def setUp(self):
        self.tpl = agent_template_map['qwen3_5']()
        self.tool_call_msg = {
            'role': 'tool_call',
            'content': json.dumps({
                'name': 'search',
                'arguments': {
                    'query': 'stock'
                },
            }),
        }
        self.tool_content = self.tpl._format_tool_calls([self.tool_call_msg])

    def test_think_plus_post_text_then_tool_call_keeps_thinking_and_separator(self):
        pre = {
            'role': 'assistant',
            'content': '<think>\nI need to check the stock first.\n</think>\n\n',
        }
        out = self.tpl._add_tool_call_prefix(self.tool_content, pre)
        # Reasoning must be preserved verbatim (this was the regression).
        self.assertIn('<think>\nI need to check the stock first.\n</think>', out)
        # Effective content after </think> is empty here, but the original
        # content ends in '\n\n' so we still want the <tool_call> to follow it
        # without an extra blank line inserted by the prefix hook.
        self.assertTrue(out.startswith(pre['content']))
        # The tool_call block must follow.
        self.assertTrue(out.endswith(self.tool_content))

    def test_pure_thinking_then_tool_call_keeps_thinking_without_extra_separator(self):
        pre = {
            'role': 'assistant',
            'content': '<think>\nonly thinking, no post-text\n</think>',
        }
        out = self.tpl._add_tool_call_prefix(self.tool_content, pre)
        # Reasoning preserved.
        self.assertIn('only thinking, no post-text', out)
        # No inserted '\\n\\n' separator between </think> and <tool_call> when
        # there is no post-think text — the jinja template just concatenates.
        self.assertNotIn('</think>\n\n<​tool_call>', out)
        self.assertTrue(out.endswith('</think>' + self.tool_content))

    def test_think_plus_post_text_then_tool_call_inserts_separator(self):
        pre = {
            'role': 'assistant',
            'content': '<think>\nplan\n</think>\nSome preamble text.',
        }
        out = self.tpl._add_tool_call_prefix(self.tool_content, pre)
        # Reasoning preserved.
        self.assertIn('plan', out)
        # Separator before tool_call when effective content is non-empty.
        self.assertIn('Some preamble text.\n\n<​tool_call>', out)

    def test_no_pre_message_returns_tool_content_unchanged(self):
        out = self.tpl._add_tool_call_prefix(self.tool_content, None)
        self.assertEqual(out, self.tool_content)

    def test_non_assistant_pre_message_returns_tool_content_unchanged(self):
        out = self.tpl._add_tool_call_prefix(self.tool_content, {
            'role': 'user',
            'content': 'no preceding assistant here',
        })
        self.assertEqual(out, self.tool_content)

    def test_empty_string_content_returns_tool_content_unchanged(self):
        out = self.tpl._add_tool_call_prefix(self.tool_content, {
            'role': 'assistant',
            'content': '',
        })
        self.assertEqual(out, self.tool_content)


class TestGetLastUserRoundIncludeTool(unittest.TestCase):
    """`_remove_history_thinking` and `_add_non_thinking_prefix` in
    ``swift.template.base`` must use the last *user* message as the boundary
    for stripping historical reasoning (not the last user-or-tool), to match
    the official Qwen3.5/3.6 chat_template.jinja. The function accepts an
    ``include_tool`` keyword since #9923; #10255 fixes the two call sites.
    """

    def test_include_tool_false_ignores_tool_messages(self):
        messages = [
            {'role': 'user', 'content': 'first user'},
            {'role': 'assistant', 'content': 'first assistant'},
            {'role': 'tool', 'content': 'tool result'},
            {'role': 'assistant', 'content': 'second assistant'},
        ]
        self.assertEqual(get_last_user_round(messages, include_tool=False), 0)
        self.assertEqual(get_last_user_round(messages, include_tool=True), 2)

    def test_include_tool_false_matches_first_user(self):
        messages = [
            {'role': 'user', 'content': 'only user'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'tool', 'content': 't1'},
            {'role': 'assistant', 'content': 'a2'},
            {'role': 'tool', 'content': 't2'},
            {'role': 'assistant', 'content': 'a3'},
        ]
        # The last user message is at index 0; everything from index 1
        # onward (including all tool messages and assistants after them)
        # belongs to the current round, so reasoning there must be
        # preserved by `_remove_history_thinking`.
        self.assertEqual(get_last_user_round(messages, include_tool=False), 0)

    def test_template_history_thinking_calls_use_include_tool_false(self):
        """Static guard so a future refactor of the two call sites cannot
        silently regress to the old (include_tool=True default) behaviour."""
        import inspect
        for name in ('_remove_history_thinking', '_add_non_thinking_prefix'):
            method = getattr(Template, name)
            source = inspect.getsource(method)
            self.assertIn(
                'get_last_user_round(messages, include_tool=False)',
                source,
                msg=(f'{name} must call '
                     '`get_last_user_round(messages, include_tool=False)` '
                     'to match the official jinja boundary; #10255'),
            )


if __name__ == '__main__':
    unittest.main()

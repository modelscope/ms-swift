# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for #10255.

The Qwen3.5/3.6 swift backend was dropping the assistant's ``<think>...</think>``
reasoning whenever the same assistant turn also carried ``tool_calls`` (and, in
the history-thinking path, whenever a tool message sat between the user prompt
and the assistant reasoning). The official ``chat_template.jinja`` (used by
vLLM / transformers at inference) keeps both, which is the root cause of the
train/infer mismatch on Qwen3.5/3.6 agent data (the same upstream issue as
#9234 — ``</think><|im_end|>`` / empty ``<tool_call>`` after SFT).

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
    ``<think>...</think>`` reasoning) before ``<tool_call>`` and only inserts
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
        self.assertNotIn('</think>\n\n<tool_call>', out)
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
        self.assertIn('Some preamble text.\n\n<tool_call>', out)

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
            {
                'role': 'user',
                'content': 'first user'
            },
            {
                'role': 'assistant',
                'content': 'first assistant'
            },
            {
                'role': 'tool',
                'content': 'tool result'
            },
            {
                'role': 'assistant',
                'content': 'second assistant'
            },
        ]
        self.assertEqual(get_last_user_round(messages, include_tool=False), 0)
        self.assertEqual(get_last_user_round(messages, include_tool=True), 2)

    def test_include_tool_false_matches_first_user(self):
        messages = [
            {
                'role': 'user',
                'content': 'only user'
            },
            {
                'role': 'assistant',
                'content': 'a1'
            },
            {
                'role': 'tool',
                'content': 't1'
            },
            {
                'role': 'assistant',
                'content': 'a2'
            },
            {
                'role': 'tool',
                'content': 't2'
            },
            {
                'role': 'assistant',
                'content': 'a3'
            },
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


class TestQwen3_5ToolCallMergesPreAssistant(unittest.TestCase):
    """`_preprocess_tool_call` must fold the preceding assistant message into
    the merged tool_call turn when the prefix re-emits its content (#10259).

    Leaving both in place renders the assistant content — reasoning
    included — twice, which is what the upstream reviewer measured against
    the official jinja (9 of 13 samples mismatched, all with the reasoning
    duplicated). Templates whose prefix emits only a separator or a channel
    token must keep the preceding message as its own turn.
    """

    def setUp(self):
        self.tpl = agent_template_map['qwen3_5']()

    def _run(self, messages, template_name='qwen3_5', pre=None):
        import swift.template.base as base
        agent = agent_template_map[template_name]()

        class _Inputs:
            pass

        inputs = _Inputs()
        inputs.tools = None
        inputs.messages = [dict(m) for m in messages]
        shim = Template.__new__(Template)
        shim._agent_template = template_name
        shim._agent_template_cache = {template_name: agent}
        shim.template_meta = None
        agent.template_meta = None
        shim._preprocess_tool_call(inputs)
        return inputs.messages

    def _tool_call_msg(self):
        return {
            'role': 'tool_call',
            'content': json.dumps({
                'name': 'search',
                'arguments': {
                    'query': 'stock'
                },
            }),
        }

    def test_think_and_tool_call_render_assistant_content_once(self):
        reasoning = '<think>\nCheck the stock quote first.\n</think>\n\nLet me look it up.'
        messages = [{
            'role': 'user',
            'content': 'How is NVDA?'
        }, {
            'role': 'assistant',
            'content': reasoning
        },
                    self._tool_call_msg(), {
                        'role': 'tool',
                        'content': 'NVDA +2%'
                    }, {
                        'role': 'assistant',
                        'content': 'NVDA is up 2 percent.'
                    }]

        out = self._run(messages)

        roles = [m['role'] for m in out]
        self.assertEqual(len(out), 4, msg=f'expected user + merged assistant + tool + answer, got {roles}')
        merged = out[1]
        self.assertEqual(merged['role'], 'assistant')
        count = merged['content'].count(reasoning)
        self.assertEqual(
            count,
            1,
            msg=(f'assistant content (with reasoning) must render exactly once; got '
                 f'{count} occurrences: {merged["content"]!r}'),
        )
        self.assertIn('<tool_call>', merged['content'])
        # The separator stays when effective post-think text exists.
        self.assertIn('\n\n<tool_call>', merged['content'])
        # The reasoning lands before the tool call, matching the jinja.
        self.assertLess(merged['content'].index(reasoning), merged['content'].index('<tool_call>'))

    def test_pure_reasoning_turn_keeps_reasoning_without_separator(self):
        reasoning = '<think>\nThe input is complete; call the tool directly.\n</think>'
        messages = [{
            'role': 'user',
            'content': 'Search the stock'
        }, {
            'role': 'assistant',
            'content': reasoning
        },
                    self._tool_call_msg()]

        out = self._run(messages)

        self.assertEqual([m['role'] for m in out], ['user', 'assistant'])
        self.assertEqual(out[1]['content'].count(reasoning), 1)
        self.assertIn('<tool_call>', out[1]['content'])
        self.assertNotIn(
            '\n\n<tool_call>',
            out[1]['content'],
            msg='a pure-reasoning turn emits the tool_call directly after, without \\n\\n',
        )

    def test_absorbed_pre_message_loss_fields_carry_over(self):
        messages = [{
            'role': 'user',
            'content': 'Search'
        }, {
            'role': 'assistant',
            'content': 'Let me search.',
            'loss': 1,
            'loss_scale': 'default'
        },
                    self._tool_call_msg(), {
                        'role': 'tool',
                        'content': 'NVDA +2%'
                    }]

        out = self._run(messages)

        merged = out[1]
        self.assertEqual(len(out), 3)
        self.assertEqual(merged.get('loss'), 1, msg='the folded assistant turn keeps its own loss field')
        self.assertEqual(merged.get('loss_scale'), 'default')

    def test_prefix_only_templates_keep_pre_message_as_own_turn(self):
        for name in ('deepseek_v4', 'kimi_k3'):
            messages = [{
                'role': 'user',
                'content': 'Search'
            }, {
                'role': 'assistant',
                'content': 'Let me search.'
            },
                        self._tool_call_msg(), {
                            'role': 'tool',
                            'content': 'NVDA +2%'
                        }]

            out = self._run(messages, template_name=name)

            self.assertEqual(
                [m['role'] for m in out],
                ['user', 'assistant', 'assistant', 'tool'],
                msg=(f'{name} uses a prefix-only hook: the preceding assistant message must '
                     'survive as its own turn'),
            )
            self.assertNotEqual(
                out[1]['content'],
                out[2]['content'],
                msg=f'{name} must not duplicate the assistant content into the tool turn',
            )

    def test_absorbs_flag_matches_prefix_consumption(self):
        template = agent_template_map['qwen3_5']()
        cases = [
            (None, False),
            ({
                'role': 'user',
                'content': 'hi'
            }, False),
            ({
                'role': 'assistant',
                'content': ''
            }, False),
            ({
                'role': 'assistant',
                'content': 'text'
            }, True),
            ({
                'role': 'assistant',
                'content': ['not', 'a', 'string']
            }, False),
        ]
        for pre_message, expected in cases:
            with self.subTest(pre_message=pre_message):
                self.assertEqual(template._tool_call_prefix_absorbs_content(pre_message), expected)
        # The graft is generic; only qwen3_5 opts in on these templates.
        for name in ('deepseek_v4', 'kimi_k3', 'qwen3_coder'):
            agent = agent_template_map[name]()
            self.assertFalse(
                agent._tool_call_prefix_absorbs_content({
                    'role': 'assistant',
                    'content': 'text'
                }),
                msg=f'{name} must keep the default (no absorption)',
            )


if __name__ == '__main__':
    unittest.main()

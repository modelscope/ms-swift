# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for #10216.

``normalize_openai_tool_calls`` used to do a ``json.loads(arguments)`` ->
``json.dumps(content)`` round-trip on every assistant ``tool_calls`` entry.
For JSON-encoded numeric arguments this silently rewrites IEEE-754 floats —
e.g. ``"0.7"`` becomes ``"0.7000000000000001"`` — which is a different token
sequence for function-call SFT.

The fix preserves the original ``arguments`` representation end-to-end at the
loader boundary. Downstream consumers (e.g. ``BaseAgentTemplate._parse_tool_call``)
already parse the JSON themselves before consuming the value, so this is a
strict improvement with no behavioural change for non-numeric payloads.
"""
import json
import unittest

from swift.template.template_inputs import StdTemplateInputs, normalize_openai_tool_calls


class TestNormalizeOpenAIToolCallsPreservesArguments(unittest.TestCase):
    """``normalize_openai_tool_calls`` must NOT ``json.loads`` the
    ``arguments`` field. The original string is the only lossless representation
    of values like ``0.7`` / ``0.1 + 0.2`` that lose precision when round-tripped
    through Python ``float`` -> ``json.dumps``.
    """

    def test_arguments_string_is_preserved_verbatim(self):
        messages = [{
            'role': 'assistant',
            'tool_calls': [{
                'function': {
                    'name': 'set_zoom',
                    'arguments': '{"zoom": 0.7}',
                },
            }],
        }]
        normalized = normalize_openai_tool_calls(messages)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]['role'], 'tool_call')
        content = normalized[0]['content']
        self.assertIsInstance(content['arguments'], str)
        self.assertEqual(content['arguments'], '{"zoom": 0.7}')

    def test_arguments_string_is_preserved_for_multiple_calls(self):
        messages = [{
            'role':
            'assistant',
            'tool_calls': [
                {
                    'function': {
                        'name': 'a',
                        'arguments': '{"x": 0.1, "y": 0.2}'
                    }
                },
                {
                    'function': {
                        'name': 'b',
                        'arguments': '{"x": 0.7}'
                    }
                },
            ],
        }]
        normalized = normalize_openai_tool_calls(messages)
        self.assertEqual(len(normalized), 2)
        for original_call, normalized_msg in zip(messages[0]['tool_calls'], normalized):
            self.assertIsInstance(normalized_msg['content']['arguments'], str)
            self.assertEqual(
                normalized_msg['content']['arguments'],
                original_call['function']['arguments'],
            )

    def test_arguments_dict_is_passed_through_unchanged(self):
        # Some loaders (e.g. parquet via pyarrow) hand us a dict directly.
        # We should not transform it; the float precision is already lost by
        # that point and downstream serialization is what it is. This test
        # documents the current pass-through behaviour and locks it in.
        arguments = {'zoom': 0.7}
        messages = [{
            'role': 'assistant',
            'tool_calls': [{
                'function': {
                    'name': 'set_zoom',
                    'arguments': arguments,
                },
            }],
        }]
        normalized = normalize_openai_tool_calls(messages)
        self.assertIs(normalized[0]['content']['arguments'], arguments)

    def test_arguments_without_function_wrapper(self):
        # Some payloads omit the ``function`` wrapper; the legacy fallback
        # already handled this — make sure the fix did not regress it.
        messages = [{
            'role': 'assistant',
            'tool_calls': [{
                'name': 'set_zoom',
                'arguments': '{"zoom": 0.7}',
            }],
        }]
        normalized = normalize_openai_tool_calls(messages)
        self.assertEqual(normalized[0]['content']['arguments'], '{"zoom": 0.7}')

    def test_assistant_without_tool_calls_is_passthrough(self):
        messages = [{
            'role': 'assistant',
            'content': 'hi',
        }]
        normalized = normalize_openai_tool_calls(messages)
        self.assertEqual(normalized, messages)


class TestStdTemplateInputsFromDictPreservesToolCallArguments(unittest.TestCase):
    """``StdTemplateInputs.from_dict`` is the actual entry point hit by the
    dataset loader. After the JSON-content re-serialization, the original
    ``arguments`` string must still be recoverable verbatim — otherwise
    function-call SFT sees a different token sequence than what is in the
    source JSONL.
    """

    def test_round_trip_preserves_arguments_string(self):
        original_arguments = '{"zoom": 0.7}'
        inputs = {
            'messages': [{
                'role': 'user',
                'content': 'Set zoom to 0.7.',
            }, {
                'role': 'assistant',
                'tool_calls': [{
                    'function': {
                        'name': 'set_zoom',
                        'arguments': original_arguments,
                    },
                }],
            }],
        }
        std_inputs = StdTemplateInputs.from_dict(inputs)
        tool_call_msgs = [m for m in std_inputs.messages if m['role'] == 'tool_call']
        self.assertEqual(len(tool_call_msgs), 1)
        # Content was re-serialized via json.dumps in from_dict, so it is now
        # a string. Parse it back and assert the inner arguments string
        # survived verbatim.
        content_str = tool_call_msgs[0]['content']
        self.assertIsInstance(content_str, str)
        content = json.loads(content_str)
        self.assertEqual(content['arguments'], original_arguments)
        self.assertEqual(content['name'], 'set_zoom')

    def test_round_trip_preserves_multiple_float_arguments(self):
        inputs = {
            'messages': [{
                'role':
                'assistant',
                'tool_calls': [{
                    'function': {
                        'name': 'configure',
                        'arguments': '{"x": 0.1, "y": 0.2, "z": 0.7}',
                    },
                }],
            }],
        }
        std_inputs = StdTemplateInputs.from_dict(inputs)
        tool_call_msg = next(m for m in std_inputs.messages if m['role'] == 'tool_call')
        content = json.loads(tool_call_msg['content'])
        self.assertEqual(content['arguments'], '{"x": 0.1, "y": 0.2, "z": 0.7}')


if __name__ == '__main__':
    unittest.main()

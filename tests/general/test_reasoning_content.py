# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import tempfile
import unittest
from pathlib import Path

from swift.dataset import load_dataset
from swift.template.template_inputs import StdTemplateInputs, normalize_openai_tool_calls


class TestReasoningContent(unittest.TestCase):

    def test_dataset_to_template_inputs(self):
        reason = '  中文🙂 quote <think> and </think>\n\n'
        messages = [
            {
                'role': 'user',
                'content': 'question'
            },
            {
                'role': 'assistant',
                'content': 'answer',
                'reasoning_content': reason,
                'loss': False
            },
            {
                'role': 'user',
                'content': 'next'
            },
            {
                'role': 'assistant',
                'content': 'final'
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'messages.jsonl'
            path.write_text(json.dumps({'messages': messages}, ensure_ascii=False) + '\n')
            dataset, _ = load_dataset([str(path)], shuffle=False, split_dataset_ratio=0, strict=True)
            inputs = StdTemplateInputs.from_dict(dataset[0])
        self.assertEqual(inputs.messages, messages)

    def test_tool_calls_keep_reasoning_once(self):
        for content in ('body', '', None):
            for reason in ('  reasoning </think>\n', ''):
                with self.subTest(content=content, reasoning=reason):
                    message = {
                        'role':
                        'assistant',
                        'content':
                        content,
                        'reasoning_content':
                        reason,
                        'loss':
                        False,
                        'loss_scale':
                        0.5,
                        'tool_calls': [
                            {
                                'function': {
                                    'name': 'edit',
                                    'arguments': {
                                        'new_str': ''
                                    }
                                }
                            },
                            {
                                'function': {
                                    'name': 'unknown',
                                    'arguments': {
                                        'value': ' 中文🙂\n '
                                    }
                                }
                            },
                        ],
                    }
                    original = copy.deepcopy(message)
                    result = normalize_openai_tool_calls([message])
                    self.assertEqual(message, original)
                    self.assertEqual(len(result), 3)
                    self.assertEqual(
                        result[0], {
                            'role': 'assistant',
                            'content': content or '',
                            'reasoning_content': reason,
                            'loss': False,
                            'loss_scale': 0.5,
                        })
                    for call, source in zip(result[1:], message['tool_calls']):
                        self.assertEqual(call['content'], source['function'])
                        self.assertNotIn('reasoning_content', call)
                        self.assertIs(call['loss'], False)
                        self.assertEqual(call['loss_scale'], 0.5)
                    self.assertEqual(normalize_openai_tool_calls(result), result)

    def test_tool_reasoning_through_dataset_loader(self):
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'edit'
                },
                {
                    'role': 'assistant',
                    'reasoning_content': 'THOUGHT',
                    'content': None,
                    'tool_calls': [{
                        'function': {
                            'name': 'edit',
                            'arguments': {
                                'new_str': ''
                            }
                        }
                    }]
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tools.jsonl'
            path.write_text(json.dumps(row) + '\n')
            dataset, _ = load_dataset([str(path)], shuffle=False, split_dataset_ratio=0, strict=True)
            inputs = StdTemplateInputs.from_dict(dataset[0])
        self.assertEqual(inputs.messages[1]['reasoning_content'], 'THOUGHT')
        self.assertEqual(inputs.messages[1]['content'], '')
        self.assertEqual(json.loads(inputs.messages[2]['content'])['arguments'], {'new_str': ''})

    def test_absent_reasoning_keeps_existing_conversion(self):
        message = {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{
                'function': {
                    'name': 'test',
                    'arguments': '{}'
                }
            }]
        }
        expected = [{'role': 'tool_call', 'content': {'name': 'test', 'arguments': {}}}]
        self.assertEqual(normalize_openai_tool_calls([message]), expected)
        message['reasoning_content'] = None
        self.assertEqual(normalize_openai_tool_calls([message]), expected)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) ModelScope Contributors. All rights reserved.
import csv
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from swift.dataset import MessagesPreprocessor, load_dataset
from swift.dataset.preprocessor.core import default_repair_messages


class TestSerializedMessages(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        cache = patch.dict(os.environ, {'MODELSCOPE_CACHE': str(self.root / 'cache')})
        cache.start()
        self.addCleanup(cache.stop)

    def test_csv_preserves_boolean_loss(self):
        messages = [{
            'role': 'user',
            'content': 'Explain 雪.',
            'loss': False,
        }, {
            'role': 'assistant',
            'content': 'It means snow.',
            'loss': True,
        }]
        path = self.root / 'messages.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['messages'])
            writer.writeheader()
            writer.writerow({'messages': json.dumps(messages, ensure_ascii=False)})
        dataset, validation = load_dataset(str(path), strict=True, load_from_cache_file=False)
        self.assertIsNone(validation)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]['messages'], messages)

    def test_csv_does_not_drop_serialized_messages(self):
        rows = [[{'role': 'user', 'content': 'Question'}], [{'role': 'assistant', 'content': 'Answer', 'loss': False}]]
        path = self.root / 'mixed.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['messages'])
            writer.writeheader()
            writer.writerows({'messages': json.dumps(messages)} for messages in rows)
        dataset, _ = load_dataset(str(path), load_from_cache_file=False)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset['messages'], rows)

    def test_jsonl_normalizes_serialized_openai_tool_calls(self):
        arguments = {'enabled': True, 'fallback': None}
        messages = [{
            'role': 'user',
            'content': 'Configure the service.',
        }, {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{
                'type': 'function',
                'function': {
                    'name': 'configure',
                    'arguments': arguments,
                },
            }],
        }]
        path = self.root / 'messages.jsonl'
        path.write_text(json.dumps({'messages': json.dumps(messages)}) + '\n', encoding='utf-8')
        dataset, _ = load_dataset(str(path), strict=True, load_from_cache_file=False)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            dataset[0]['messages'],
            [messages[0], {
                'role': 'tool_call',
                'content': {
                    'name': 'configure',
                    'arguments': arguments,
                },
            }])

    def test_serialized_rejected_messages(self):
        rejected = [{'role': 'assistant', 'content': 'Incorrect.', 'loss': False}]
        row = {
            'messages': [{
                'role': 'assistant',
                'content': 'Correct.'
            }],
            'rejected_messages': json.dumps(rejected),
        }
        self.assertEqual(MessagesPreprocessor().preprocess(row)['rejected_messages'], rejected)

    def test_python_literal_messages_remain_supported(self):
        messages = [{
            'role': 'user',
            'content': 'Question',
            'loss': False
        }, {
            'role': 'assistant',
            'content': 'Answer',
            'loss': True
        }]
        result = MessagesPreprocessor().preprocess({'messages': repr(messages)})
        self.assertEqual(result['messages'], messages)
        self.assertEqual(default_repair_messages("[{'content': None}]"), [{'content': None}])

    def test_native_messages_are_not_copied(self):
        messages = [{'role': 'user', 'content': 'Question'}]
        self.assertIs(default_repair_messages(messages), messages)
        result = MessagesPreprocessor().preprocess({'messages': deepcopy(messages)})
        self.assertEqual(result['messages'], messages)

    def test_invalid_serialized_messages_raise(self):
        for content in ['not a message list', '[{"role": "user"}']:
            with self.subTest(content=content), self.assertRaises((ValueError, SyntaxError)):
                default_repair_messages(content)


if __name__ == '__main__':
    unittest.main()

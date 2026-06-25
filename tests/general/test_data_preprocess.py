import unittest

from swift.dataset import EncodePreprocessor, MessagesPreprocessor, PackingDataset, load_dataset
from swift.model import get_processor
from swift.template import get_template


class TestDataPreprocess(unittest.TestCase):
    """Lightweight data preprocessing tests (no model forward/backward).

    These are fast tests suitable for CI. They cover:
    - SFT dataset encode (input_ids/labels)
    - Truncation/max_length
    - Data collator padding (attention_mask)
    - Multi-turn messages
    - Tool message
    - Packing dataset

    Why these tests are needed:
    - Swift's data preprocessing pipeline is complex (template -> encode -> collate -> pack).
      NPU training failures often stem from shape/mask/label mismatches before the model
      even sees the data, not from operator issues.
    - The original tests/general/test_dataset.py and test_template.py use top-level
      functions and remote 7B models, so they are never run by unittest discovery
      and are too heavy for CI.
    """

    MODEL_PATH = 'Qwen/Qwen2-0.5B'

    @classmethod
    def setUpClass(cls):
        cls.processor = get_processor(cls.MODEL_PATH)
        cls.template = get_template(cls.processor)
        cls.template.mode = 'train'
        cls.template.init_processor(cls.processor)

    def _encode_dataset(self, dataset):
        encode_preprocessor = EncodePreprocessor(self.template)
        return encode_preprocessor(dataset, num_proc=1, load_from_cache_file=False, strict=False)

    def test_sft_dataset_encode(self):
        dataset, _ = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#20'], num_proc=1, strict=False)
        self.assertGreater(len(dataset), 0)
        encoded_dataset = self._encode_dataset(dataset)
        first = encoded_dataset[0]
        self.assertIn('input_ids', first)
        self.assertIn('labels', first)
        self.assertEqual(len(first['input_ids']), len(first['labels']))

    def test_truncation_max_length(self):
        self.template.max_length = 128
        dataset, _ = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#20'], num_proc=1, strict=False)
        encoded_dataset = self._encode_dataset(dataset)
        for row in encoded_dataset:
            self.assertLessEqual(len(row['input_ids']), self.template.max_length)
        self.template.max_length = None

    def test_data_collator_padding(self):
        dataset, _ = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#20'], num_proc=1, strict=False)
        encoded_dataset = self._encode_dataset(dataset)
        batch = [encoded_dataset[i] for i in range(4)]
        collated = self.template.data_collator(batch)
        self.assertIn('input_ids', collated)
        self.assertIn('labels', collated)
        self.assertIn('attention_mask', collated)
        self.assertEqual(collated['input_ids'].shape[0], 4)

    def test_multi_turn_messages(self):
        multi_turn_row = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'What is Python?'
                },
                {
                    'role': 'assistant',
                    'content': 'Python is a programming language.'
                },
                {
                    'role': 'user',
                    'content': 'What are its advantages?'
                },
                {
                    'role': 'assistant',
                    'content': 'Python is easy to learn and use.'
                },
            ]
        }
        encoded = self.template.encode(multi_turn_row, return_length=True)
        self.assertIn('input_ids', encoded)
        self.assertIn('labels', encoded)
        self.assertGreater(len(encoded['input_ids']), 0)
        self.assertEqual(len(encoded['input_ids']), len(encoded['labels']))

    def test_tool_message(self):
        tool_row = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'What is the weather in Beijing?'
                },
                {
                    'role':
                    'assistant',
                    'content':
                    '',
                    'tool_calls': [{
                        'type': 'function',
                        'function': {
                            'name': 'get_weather',
                            'arguments': '{"city": "Beijing"}'
                        }
                    }]
                },
                {
                    'role': 'tool',
                    'content': '{"temperature": 25, "condition": "sunny"}'
                },
                {
                    'role': 'assistant',
                    'content': 'The weather in Beijing is sunny with a temperature of 25 degrees.'
                },
            ]
        }
        encoded = self.template.encode(tool_row, return_length=True)
        self.assertIn('input_ids', encoded)
        self.assertIn('labels', encoded)
        self.assertGreater(len(encoded['input_ids']), 0)

    def test_packing_dataset(self):
        dataset, _ = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#20'], num_proc=1, strict=False)
        encoded_dataset = self._encode_dataset(dataset)
        packing_dataset = PackingDataset(
            self.template,
            encoded_dataset,
            num_proc=1,
            strict=False,
            load_from_cache_file=False,
            packing_length=512,
            packing_num_proc=1,
        )
        self.assertGreater(len(packing_dataset), 0)
        packed = packing_dataset[0]
        self.assertIsInstance(packed, list)
        self.assertGreater(len(packed), 0)
        self.assertIn('input_ids', packed[0])
        self.assertIn('labels', packed[0])


class TestRejectedMessagesPreprocess(unittest.TestCase):
    """MessagesPreprocessor handling of rejected_messages (no model required)."""

    def test_empty_rejected_messages_does_not_crash(self):
        """A DPO row whose rejected_messages repair to empty must not crash.

        The recursive preprocess() call returns None when rejected_messages is
        empty (the same graceful-skip path used for the main messages list), so
        subscripting it with ['messages'] raised TypeError and aborted the whole
        dataset map. Downstream already treats rejected_messages is None as
        'no rejected', so the row should fall back to None instead.
        """
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'Q'
                },
                {
                    'role': 'assistant',
                    'content': 'good'
                },
            ],
            'rejected_messages': [],
        }
        result = MessagesPreprocessor().preprocess(row)
        self.assertIsNotNone(result)
        self.assertIsNone(result['rejected_messages'])

    def test_valid_rejected_messages_preserved(self):
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'Q'
                },
                {
                    'role': 'assistant',
                    'content': 'good'
                },
            ],
            'rejected_messages': [
                {
                    'role': 'user',
                    'content': 'Q'
                },
                {
                    'role': 'assistant',
                    'content': 'bad'
                },
            ],
        }
        result = MessagesPreprocessor().preprocess(row)
        self.assertEqual(result['rejected_messages'][-1]['content'], 'bad')


if __name__ == '__main__':
    unittest.main()

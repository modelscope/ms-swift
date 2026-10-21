import unittest

from swift.dataset.dataset.llm import _repair_ms_bench


class TestRepairMsBench(unittest.TestCase):
    """Pure unit tests for the ms_bench messages repair function (no network)."""

    def test_empty_messages_returns_none(self):
        # An empty row can't be repaired; it must be skipped (None) like the MOSS
        # case rather than crashing the whole dataset load on messages[0].
        self.assertIsNone(_repair_ms_bench('[]'))
        self.assertIsNone(_repair_ms_bench([]))

    def test_strips_default_system_message(self):
        messages = [
            {
                'from': 'system',
                'value': 'You are a helpful assistant.'
            },
            {
                'from': 'user',
                'value': 'hi'
            },
        ]
        self.assertEqual(_repair_ms_bench(messages), [{'from': 'user', 'value': 'hi'}])

    def test_keeps_a_normal_conversation(self):
        messages = [
            {
                'from': 'user',
                'value': 'hi'
            },
            {
                'from': 'assistant',
                'value': 'hello'
            },
        ]
        self.assertEqual(_repair_ms_bench(messages), messages)

    def test_skips_moss_rows(self):
        self.assertIsNone(_repair_ms_bench([{'from': 'user', 'value': 'moss reply'}]))


if __name__ == '__main__':
    unittest.main()

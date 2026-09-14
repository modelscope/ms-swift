# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from unittest.mock import patch

from swift.infer_engine import AdapterRequest, TransformersEngine


class TestTransformersAdapterCache(unittest.TestCase):

    def setUp(self):
        self.engine = object.__new__(TransformersEngine)
        self.engine._adapters_pool = {}

    def test_failed_load_can_be_retried(self):
        adapter = AdapterRequest('candidate', '/adapters/candidate')
        error = ValueError('invalid adapter config')
        with patch.object(self.engine, '_add_adapter', side_effect=[error, None]) as load:
            with self.assertRaises(ValueError) as context:
                self.engine._get_adapter_names(adapter)
            self.assertIs(context.exception, error)
            self.assertEqual(self.engine._adapters_pool, {})
            self.assertIsNone(self.engine._get_adapter_names(None))
            self.assertEqual(self.engine._get_adapter_names(adapter), ['candidate'])
            self.assertEqual(self.engine._adapters_pool, {'candidate': adapter})
            self.assertEqual(self.engine._get_adapter_names(adapter), ['candidate'])
            self.assertEqual(self.engine._get_adapter_names(None), ['__base__'])
            self.assertEqual(load.call_count, 2)
            load.assert_called_with(adapter.path, adapter.name)

    def test_failed_load_preserves_existing_adapters(self):
        existing = AdapterRequest('existing', '/adapters/existing')
        candidate = AdapterRequest('candidate', '/adapters/candidate')
        with patch.object(self.engine, '_add_adapter') as load:
            self.assertEqual(self.engine._get_adapter_names(existing), ['existing'])
            load.side_effect = OSError('adapter unavailable')
            with self.assertRaisesRegex(OSError, 'adapter unavailable'):
                self.engine._get_adapter_names(candidate)
            self.assertEqual(self.engine._adapters_pool, {'existing': existing})
            self.assertEqual(self.engine._get_adapter_names(existing), ['existing'])
            self.assertEqual(self.engine._get_adapter_names(None), ['__base__'])
            self.assertEqual(load.call_count, 2)


if __name__ == '__main__':
    unittest.main()

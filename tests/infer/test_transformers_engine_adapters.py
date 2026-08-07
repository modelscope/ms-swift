import unittest
from unittest.mock import Mock

from swift.infer_engine.transformers_engine import TransformersEngine
from swift.infer_engine.utils import AdapterRequest


class TestTransformersEngineAdapters(unittest.TestCase):

    def setUp(self):
        self.engine = object.__new__(TransformersEngine)
        self.engine._adapters_pool = {}
        self.engine._add_adapter = Mock()

    def test_adapter_name_matches_batch_size(self):
        adapter_request = AdapterRequest('lora1', '/tmp/lora1')

        adapter_names = self.engine._get_adapter_names(adapter_request, batch_size=3)

        self.assertEqual(adapter_names, ['lora1', 'lora1', 'lora1'])
        self.engine._add_adapter.assert_called_once_with('/tmp/lora1', 'lora1')

    def test_base_adapter_name_matches_batch_size(self):
        self.engine._adapters_pool['lora1'] = AdapterRequest('lora1', '/tmp/lora1')

        adapter_names = self.engine._get_adapter_names(None, batch_size=2)

        self.assertEqual(adapter_names, ['__base__', '__base__'])


if __name__ == '__main__':
    unittest.main()

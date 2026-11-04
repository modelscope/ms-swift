import unittest
from unittest.mock import Mock, call, patch

from swift.utils import torch_utils

GIB = 1024**3


class TestMaxReservedMemory(unittest.TestCase):

    @patch.object(torch_utils, 'get_device_count', return_value=3)
    @patch.object(torch_utils, 'is_mp', return_value=True)
    def test_model_parallel_reports_per_device_maximum(self, _, __):
        device_api = Mock()
        device_api.max_memory_reserved.side_effect = [16 * GIB, 40 * GIB, 24 * GIB]

        with patch.object(torch_utils, 'get_torch_device', return_value=device_api):
            memory = torch_utils.get_max_reserved_memory()

        self.assertEqual(memory, 40)
        self.assertEqual(
            device_api.max_memory_reserved.call_args_list,
            [call(device=0), call(device=1), call(device=2)],
        )

    @patch.object(torch_utils, 'get_device_count')
    @patch.object(torch_utils, 'is_mp', return_value=False)
    def test_non_model_parallel_uses_current_device(self, _, get_device_count):
        device_api = Mock()
        device_api.max_memory_reserved.return_value = 12 * GIB

        with patch.object(torch_utils, 'get_torch_device', return_value=device_api):
            memory = torch_utils.get_max_reserved_memory()

        self.assertEqual(memory, 12)
        get_device_count.assert_not_called()
        device_api.max_memory_reserved.assert_called_once_with(device=None)

    @patch.object(torch_utils, 'is_mp', return_value=False)
    def test_missing_memory_api_returns_zero(self, _):
        with patch.object(torch_utils, 'get_torch_device', return_value=object()):
            memory = torch_utils.get_max_reserved_memory()

        self.assertEqual(memory, 0)


if __name__ == '__main__':
    unittest.main()

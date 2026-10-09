# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from unittest.mock import Mock, patch

from swift.model import patcher

GIB = 1024**3


class TestMpDdpMaxMemory(unittest.TestCase):
    """`_get_max_memory` must query the active accelerator, not a pinned CUDA namespace."""

    @patch.object(patcher, 'get_device_count', return_value=1)
    def test_resolves_memory_through_the_active_accelerator(self, _):
        device_api = Mock()
        device_api.mem_get_info.return_value = (8 * GIB, 16 * GIB)

        with patch.object(patcher, 'get_device', return_value=torch.device('cpu')), \
                patch.object(patcher, 'get_torch_device', return_value=device_api):
            max_memory = patcher._get_max_memory([0])

        self.assertEqual(max_memory[0], 8 * GIB)
        device_api.mem_get_info.assert_called_once_with(0)

    @patch.object(patcher, 'get_device_count', return_value=2)
    def test_devices_outside_the_shard_report_zero(self, _):
        device_api = Mock()
        device_api.mem_get_info.return_value = (4 * GIB, 8 * GIB)

        with patch.object(patcher, 'get_device', return_value=torch.device('cpu')), \
                patch.object(patcher, 'get_torch_device', return_value=device_api):
            max_memory = patcher._get_max_memory([1])

        self.assertEqual(max_memory[0], 0)
        self.assertEqual(max_memory[1], 4 * GIB)
        device_api.mem_get_info.assert_called_once_with(1)

    @patch.object(patcher, 'get_device_count', return_value=1)
    def test_reports_available_cpu_memory(self, _):
        device_api = Mock()
        device_api.mem_get_info.return_value = (1 * GIB, 2 * GIB)

        with patch.object(patcher, 'get_device', return_value=torch.device('cpu')), \
                patch.object(patcher, 'get_torch_device', return_value=device_api):
            max_memory = patcher._get_max_memory([0])

        self.assertGreater(max_memory['cpu'], 0)


if __name__ == '__main__':
    unittest.main()

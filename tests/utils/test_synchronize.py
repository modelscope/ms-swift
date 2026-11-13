# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swift.utils import torch_utils


class TestSynchronize(unittest.TestCase):

    def setUp(self):
        self.available = {}
        for backend in ('cuda', 'npu', 'musa', 'mps'):
            patcher = patch.object(torch_utils, f'is_torch_{backend}_available', return_value=False)
            self.available[backend] = patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(torch.cuda, 'synchronize')
        self.cuda_sync = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(torch.mps, 'synchronize')
        self.mps_sync = patcher.start()
        self.addCleanup(patcher.stop)

    def test_cpu_does_not_call_accelerator(self):
        torch_utils.synchronize()
        self.cuda_sync.assert_not_called()
        self.mps_sync.assert_not_called()

    def test_explicit_cpu_with_accelerator_available(self):
        self.available['cuda'].return_value = True
        for device in ('cpu', torch.device('cpu')):
            torch_utils.synchronize(device)
        self.cuda_sync.assert_not_called()
        self.mps_sync.assert_not_called()

    def test_default_mps_waits_for_device(self):
        self.available['mps'].return_value = True
        torch_utils.synchronize()
        self.mps_sync.assert_called_once_with()
        self.cuda_sync.assert_not_called()

    def test_explicit_mps_does_not_pass_device_argument(self):
        for device in ('mps', torch.device('mps')):
            with self.subTest(device=device):
                self.mps_sync.reset_mock()
                torch_utils.synchronize(device)
                self.mps_sync.assert_called_once_with()
        self.cuda_sync.assert_not_called()

    def test_cuda_preserves_device_argument(self):
        self.available['cuda'].return_value = True
        torch_utils.synchronize(2)
        self.cuda_sync.assert_called_once_with(2)

    def test_explicit_cuda_does_not_fall_back_to_mps(self):
        self.available['mps'].return_value = True
        self.cuda_sync.side_effect = RuntimeError('CUDA is unavailable')
        with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
            torch_utils.synchronize('cuda:0')
        self.cuda_sync.assert_called_once_with('cuda:0')
        self.mps_sync.assert_not_called()

    def test_npu_preserves_device_argument(self):
        self.available['npu'].return_value = True
        npu = SimpleNamespace(synchronize=Mock())
        with patch.object(torch, 'npu', npu, create=True):
            torch_utils.synchronize(1)
        npu.synchronize.assert_called_once_with(1)
        self.cuda_sync.assert_not_called()

    def test_musa_still_takes_precedence_over_cuda(self):
        self.available['musa'].return_value = True
        self.available['cuda'].return_value = True
        musa = SimpleNamespace(synchronize=Mock())
        with patch.object(torch, 'musa', musa, create=True):
            torch_utils.synchronize(3)
        musa.synchronize.assert_called_once_with(3)
        self.cuda_sync.assert_not_called()

    def test_cpu_timer_returns_monotonic_timestamps(self):
        first = torch_utils.time_synchronize()
        second = torch_utils.time_synchronize()
        self.assertIsInstance(first, float)
        self.assertGreaterEqual(second, first)
        self.cuda_sync.assert_not_called()


class TestSynchronizeOnDevice(unittest.TestCase):

    def test_actual_cpu(self):
        torch_utils.synchronize(torch.device('cpu'))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS is required')
    def test_actual_mps(self):
        values = torch.arange(32, dtype=torch.float32, device='mps').square()
        torch_utils.synchronize(torch.device('mps'))
        torch.testing.assert_close(values.cpu(), torch.arange(32, dtype=torch.float32).square())
        self.assertIsInstance(torch_utils.time_synchronize(), float)


if __name__ == '__main__':
    unittest.main()

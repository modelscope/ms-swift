import os
import subprocess
import sys
import tempfile
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from swift.utils import torch_utils
from swift.utils.import_utils import is_torch_musa_installed, is_torchada_available


def _fake_musa(bf16=True):
    calls = []
    return SimpleNamespace(
        calls=calls,
        synchronize=lambda device=None: calls.append(('synchronize', device)),
        current_device=lambda: 3,
        set_device=lambda index: calls.append(('set_device', index)),
        device_count=lambda: 8,
        empty_cache=lambda: calls.append(('empty_cache', )),
        ipc_collect=lambda: calls.append(('ipc_collect', )),
        is_bf16_supported=lambda: bf16,
        is_available=lambda: True,
    )


class TestMusaImportUtils(unittest.TestCase):

    def test_installed_follows_torch_musa_spec(self):
        with patch('importlib.util.find_spec', side_effect=lambda name: object() if name == 'torch_musa' else None):
            self.assertTrue(is_torch_musa_installed())
            self.assertFalse(is_torchada_available())
        with patch('importlib.util.find_spec', side_effect=lambda name: object() if name == 'torchada' else None):
            self.assertFalse(is_torch_musa_installed())
            self.assertTrue(is_torchada_available())


class TestTorchadaBootstrap(unittest.TestCase):
    """`import swift` must import torchada first, but only when torch_musa is installed."""

    @staticmethod
    def _torchada_imported(torch_musa, torchada):
        """Run `import swift` in a clean interpreter with stub torch_musa/torchada packages (or none)."""
        with tempfile.TemporaryDirectory() as tmp:
            hidden = []
            for name, present in [('torch_musa', torch_musa), ('torchada', torchada)]:
                if present:
                    (Path(tmp) / name).mkdir()
                    (Path(tmp) / name / '__init__.py').write_text('')
                else:
                    hidden.append(name)  # hide a real install: find_spec() returns None for sys.modules[name] = None
            code = (f'import sys; sys.modules.update(dict.fromkeys({hidden!r})); '
                    'import swift; print(sys.modules.get("torchada") is not None)')
            root = Path(__file__).resolve().parents[2]
            env = {**os.environ, 'PYTHONPATH': os.pathsep.join([tmp, str(root), os.environ.get('PYTHONPATH', '')])}
            out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True)
        return out.stdout.strip().splitlines()[-1] == 'True'

    def test_imported_when_torch_musa_is_installed(self):
        self.assertTrue(self._torchada_imported(torch_musa=True, torchada=True))

    def test_skipped_without_torch_musa(self):
        self.assertFalse(self._torchada_imported(torch_musa=False, torchada=True))

    def test_skipped_without_torchada(self):
        self.assertFalse(self._torchada_imported(torch_musa=True, torchada=False))


class TestMusaDeviceHelpers(unittest.TestCase):
    """Route the helpers through a fake `torch.musa` so they run without MUSA hardware."""

    def setUp(self):
        self.fake = _fake_musa()
        patchers = [
            patch.object(torch, 'musa', self.fake, create=True),
            patch.object(torch_utils, 'is_torch_musa_available', lambda: True),
            patch.object(torch_utils, 'is_torch_npu_available', lambda: False),
            patch.object(torch_utils, 'is_torch_cuda_available', lambda: False),
            patch.object(torch_utils, 'is_torch_mps_available', lambda: False),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_get_device(self):
        self.assertEqual(torch_utils.get_device(2), 'musa:2')

    def test_current_device_and_module(self):
        self.assertEqual(torch_utils.get_current_device(), 3)
        self.assertIs(torch_utils.get_torch_device(), self.fake)

    def test_device_count(self):
        self.assertEqual(torch_utils.get_device_count(), 8)

    def test_set_device_uses_local_rank(self):
        torch_utils.set_device(5)
        self.assertEqual(self.fake.calls, [('set_device', 5)])

    def test_synchronize_empty_cache_ipc_collect(self):
        torch_utils.synchronize()
        torch_utils.empty_cache()
        torch_utils.ipc_collect()
        self.assertEqual(self.fake.calls, [('synchronize', None), ('empty_cache', ), ('ipc_collect', )])

    def test_init_process_group_defaults_to_mccl(self):
        with patch.object(torch_utils.dist, 'is_initialized', return_value=False), \
                patch.object(torch_utils.dist, 'init_process_group') as init:
            torch_utils.init_process_group()
        self.assertEqual(init.call_args.kwargs['backend'], 'mccl')

    def test_npu_takes_precedence_over_musa(self):
        with patch.object(torch_utils, 'is_torch_npu_available', lambda: True), \
                patch.object(torch, 'npu', SimpleNamespace(current_device=lambda: 0), create=True):
            self.assertEqual(torch_utils.get_device(1), 'npu:1')


class TestMusaModelDefaults(unittest.TestCase):

    def setUp(self):
        from swift.model import utils as model_utils
        self.model_utils = model_utils
        patchers = [
            patch.object(torch, 'musa', _fake_musa(), create=True),
            patch.object(model_utils, 'is_torch_musa_available', lambda: True),
            patch.object(model_utils, 'is_torch_npu_available', lambda: False),
            patch.object(model_utils, 'is_torch_cuda_available', lambda: False),
            patch.object(model_utils, 'is_torch_mps_available', lambda: False),
            patch.object(model_utils, 'is_torch_bf16_gpu_available', lambda: False),
            patch.object(model_utils, 'is_deepspeed_zero3_enabled', lambda: False),
            patch.object(model_utils, 'get_dist_setting', lambda: (0, 1, 2, 2)),
            patch.object(model_utils, 'is_mp', lambda: False),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_device_map_uses_local_rank(self):
        self.assertEqual(self.model_utils.get_default_device_map(), 'musa:1')

    def test_device_map_auto_for_model_parallel(self):
        with patch.object(self.model_utils, 'is_mp', lambda: True):
            self.assertEqual(self.model_utils.get_default_device_map(), 'auto')

    def test_default_dtype_is_bf16_when_supported(self):
        self.assertEqual(self.model_utils.get_default_torch_dtype(None), torch.bfloat16)

    def test_default_dtype_is_fp16_without_bf16(self):
        with patch.object(torch, 'musa', _fake_musa(bf16=False), create=True):
            self.assertEqual(self.model_utils.get_default_torch_dtype(None), torch.float16)

    def test_config_dtype_is_kept(self):
        self.assertEqual(self.model_utils.get_default_torch_dtype(torch.float32), torch.float32)


class TestSelectDevice(unittest.TestCase):

    def test_select_device_sets_musa_visible_devices(self):
        from swift.utils import select_device
        with patch.dict(os.environ, {}, clear=False):
            select_device('1,2')
            self.assertEqual(os.environ['MUSA_VISIBLE_DEVICES'], '1,2')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '1,2')


class TestMusaOptimizerDefault(unittest.TestCase):

    @staticmethod
    def _optim(musa, **kwargs):
        from swift.trainers import arguments
        with patch.object(arguments, 'is_torch_musa_available', lambda: musa):
            return arguments.TrainingArguments(output_dir=tempfile.gettempdir(), report_to='none', **kwargs).optim

    def test_fused_adamw_falls_back_on_musa(self):
        self.assertEqual(self._optim(True, optim='adamw_torch_fused'), 'adamw_torch')

    def test_other_optimizers_untouched_on_musa(self):
        self.assertEqual(self._optim(True, optim='adamw_torch'), 'adamw_torch')
        self.assertEqual(self._optim(True, optim='sgd'), 'sgd')

    def test_fused_adamw_kept_off_musa(self):
        self.assertEqual(self._optim(False, optim='adamw_torch_fused'), 'adamw_torch_fused')


@unittest.skipUnless(torch_utils.is_torch_musa_available(), 'requires a MUSA device')
class TestMusaHardware(unittest.TestCase):

    def test_helpers_match_torch_musa(self):
        self.assertEqual(torch_utils.get_device_count(), torch.musa.device_count())
        self.assertEqual(torch_utils.get_device(0), 'musa:0')
        self.assertEqual(torch_utils.get_torch_device().device_count(), torch.musa.device_count())

    def test_tensor_lives_on_musa_device(self):
        x = torch.ones(4, device=torch_utils.get_device(0))
        self.assertEqual(x.device.type, 'musa')
        torch_utils.synchronize()
        torch_utils.empty_cache()

    @unittest.skipUnless(is_torchada_available(), 'requires torchada')
    def test_cuda_apis_are_redirected_by_torchada(self):
        self.assertEqual(torch.device('cuda:0').type, 'musa')
        self.assertEqual(torch.cuda.device_count(), torch.musa.device_count())


if __name__ == '__main__':
    unittest.main()

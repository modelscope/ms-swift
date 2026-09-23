import os
import subprocess
import sys
import tempfile
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from swift.cli.utils import sync_musa_visible_devices, try_use_single_device_mode
from swift.utils import torch_utils
from swift.utils.import_utils import is_torch_musa_installed, is_torchada_available

_DEVICE_ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'MUSA_VISIBLE_DEVICES', 'LOCAL_RANK', 'SWIFT_SINGLE_DEVICE_MODE')

# Stub torchada that records the state it is imported in, so the tests can check what `import swift` did before it.
_STUB_TORCHADA = """import os, sys
MUSA_VISIBLE_DEVICES = os.environ.get('MUSA_VISIBLE_DEVICES')
LOCAL_RANK = os.environ.get('LOCAL_RANK')
TORCH_IMPORTED = 'torch' in sys.modules
# swift.utils imports torch when trl is installed, so it must load after torchada.
SWIFT_UTILS_IMPORTED = 'swift.utils' in sys.modules
"""


def _run_python(code, env=None, stubs=(), hidden=()):
    """Run `code` in a clean interpreter with stub packages, hidden packages and a clean device env."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, source in stubs:
            (Path(tmp) / name).mkdir()
            (Path(tmp) / name / '__init__.py').write_text(source)
        # sys.modules[name] = None makes find_spec() return None, which hides a real install.
        code = f'import sys; sys.modules.update(dict.fromkeys({list(hidden)!r})); {code}'
        root = Path(__file__).resolve().parents[2]
        base_env = {k: v for k, v in os.environ.items() if k not in _DEVICE_ENV_KEYS}
        base_env['PYTHONPATH'] = os.pathsep.join([tmp, str(root), os.environ.get('PYTHONPATH', '')])
        out = subprocess.run([sys.executable, '-c', code],
                             env={
                                 **base_env,
                                 **(env or {})
                             },
                             capture_output=True,
                             text=True,
                             check=True)
    return out.stdout.strip().splitlines()[-1]


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
        stubs = [(name, '') for name, present in [('torch_musa', torch_musa), ('torchada', torchada)] if present]
        hidden = [name for name, present in [('torch_musa', torch_musa), ('torchada', torchada)] if not present]
        code = 'import swift; print(sys.modules.get("torchada") is not None)'
        return _run_python(code, stubs=stubs, hidden=hidden) == 'True'

    def test_imported_when_torch_musa_is_installed(self):
        self.assertTrue(self._torchada_imported(torch_musa=True, torchada=True))

    def test_skipped_without_torch_musa(self):
        self.assertFalse(self._torchada_imported(torch_musa=False, torchada=True))

    def test_skipped_without_torchada(self):
        self.assertFalse(self._torchada_imported(torch_musa=True, torchada=False))

    def test_visible_devices_are_final_before_torch_is_imported(self):
        # torch autoloads torch_musa, which reads MUSA_VISIBLE_DEVICES once, so it must be set before torch loads.
        code = ('import swift, torchada; '
                'print(torchada.MUSA_VISIBLE_DEVICES, torchada.LOCAL_RANK, torchada.TORCH_IMPORTED, '
                'torchada.SWIFT_UTILS_IMPORTED)')
        env = {'CUDA_VISIBLE_DEVICES': '2,3', 'SWIFT_SINGLE_DEVICE_MODE': '1', 'LOCAL_RANK': '1'}
        out = _run_python(code, env=env, stubs=[('torch_musa', ''), ('torchada', _STUB_TORCHADA)])
        self.assertEqual(out, '3 0 False False')

    def test_device_env_untouched_without_torch_musa(self):
        code = 'import os, swift; print(os.environ.get("MUSA_VISIBLE_DEVICES"), os.environ["CUDA_VISIBLE_DEVICES"])'
        out = _run_python(code, env={'CUDA_VISIBLE_DEVICES': '2,3'}, hidden=['torch_musa'])
        self.assertEqual(out, 'None 2,3')


class TestMusaVisibleDevices(unittest.TestCase):

    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in _DEVICE_ENV_KEYS:
            os.environ.pop(key, None)

    def test_sync_copies_cuda_visible_devices(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
        sync_musa_visible_devices()
        self.assertEqual(os.environ['MUSA_VISIBLE_DEVICES'], '1,2')
        self.assertNotIn('CUDA_VISIBLE_DEVICES', os.environ)

    def test_sync_keeps_explicit_musa_visible_devices(self):
        os.environ.update(CUDA_VISIBLE_DEVICES='1,2', MUSA_VISIBLE_DEVICES='5')
        sync_musa_visible_devices()
        self.assertEqual(os.environ['MUSA_VISIBLE_DEVICES'], '5')
        self.assertNotIn('CUDA_VISIBLE_DEVICES', os.environ)

    def test_sync_is_noop_without_cuda_visible_devices(self):
        sync_musa_visible_devices()
        self.assertNotIn('MUSA_VISIBLE_DEVICES', os.environ)

    def test_single_device_mode_uses_musa_visible_devices(self):
        os.environ.update(SWIFT_SINGLE_DEVICE_MODE='1', LOCAL_RANK='1', MUSA_VISIBLE_DEVICES='4,6')
        try_use_single_device_mode()
        self.assertEqual((os.environ['MUSA_VISIBLE_DEVICES'], os.environ['LOCAL_RANK']), ('6', '0'))
        try_use_single_device_mode()  # idempotent: the CLI entry points call it again after `import swift`
        self.assertEqual((os.environ['MUSA_VISIBLE_DEVICES'], os.environ['LOCAL_RANK']), ('6', '0'))

    def test_single_device_mode_on_cuda_is_unchanged(self):
        os.environ.update(SWIFT_SINGLE_DEVICE_MODE='1', LOCAL_RANK='1', CUDA_VISIBLE_DEVICES='4,6')
        try_use_single_device_mode()
        self.assertEqual((os.environ['CUDA_VISIBLE_DEVICES'], os.environ['LOCAL_RANK']), ('6', '0'))
        self.assertNotIn('MUSA_VISIBLE_DEVICES', os.environ)


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

    def test_musa_takes_precedence_over_patched_cuda(self):
        # MUSA patches such as megatron-lm-musa-patch make torch.cuda.is_available() return True.
        with patch.object(torch_utils, 'is_torch_cuda_available', lambda: True):
            self.assertIs(torch_utils.get_torch_device(), self.fake)
            torch_utils.ipc_collect()
        self.assertEqual(self.fake.calls, [('ipc_collect', )])

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


class TestMusaActivationOffload(unittest.TestCase):

    def test_offload_streams_use_musa(self):
        from swift.callbacks import activation_cpu_offload as offload
        fake = _fake_musa()
        with patch.object(offload, 'is_cuda_available', False), patch.object(offload, 'is_npu_available', False), \
                patch.object(offload, 'is_musa_available', True), patch.object(torch, 'musa', fake, create=True):
            self.assertEqual(offload.get_device_name(), 'musa')
            self.assertIs(offload.get_torch_device(), fake)

    def test_musa_takes_precedence_over_patched_cuda(self):
        from swift.callbacks import activation_cpu_offload as offload
        with patch.object(offload, 'is_cuda_available', True), patch.object(offload, 'is_npu_available', False), \
                patch.object(offload, 'is_musa_available', True):
            self.assertEqual(offload.get_device_name(), 'musa')


class TestMusaLoadStateFile(unittest.TestCase):

    def test_musa_takes_precedence_over_patched_cuda(self):
        from swift.tuners import base
        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, base.SAFETENSORS_WEIGHTS_NAME).touch()
            with patch.object(base, 'is_torch_musa_available', lambda: True), \
                    patch.object(torch.cuda, 'is_available', lambda: True), \
                    patch('safetensors.torch.load_file') as load_file:
                base.SwiftModel.load_state_file(tmp_dir)
        self.assertEqual(load_file.call_args.kwargs['device'], 'musa')


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

    @unittest.skipUnless(is_torchada_available(), 'requires torchada')
    def test_cuda_visible_devices_limits_musa_devices(self):
        code = ('import os, swift, torch; print(torch.musa.device_count(), os.environ["MUSA_VISIBLE_DEVICES"], '
                '"CUDA_VISIBLE_DEVICES" in os.environ)')
        self.assertEqual(_run_python(code, env={'CUDA_VISIBLE_DEVICES': '1'}), '1 1 False')

    @unittest.skipUnless(is_torchada_available(), 'requires torchada')
    def test_single_device_mode_limits_musa_devices(self):
        code = 'import os, swift, torch; print(torch.musa.device_count(), os.environ["LOCAL_RANK"])'
        env = {'CUDA_VISIBLE_DEVICES': '0,1', 'SWIFT_SINGLE_DEVICE_MODE': '1', 'LOCAL_RANK': '1'}
        self.assertEqual(_run_python(code, env=env), '1 0')


if __name__ == '__main__':
    unittest.main()

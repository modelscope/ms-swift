# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'examples' / 'ascend' / 'check_env.py'
spec = importlib.util.spec_from_file_location('ascend_check_env', SCRIPT)
check_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_env)


class TestAscendCheckEnv(unittest.TestCase):

    def test_report_continues_when_packages_missing(self):
        output = io.StringIO()
        with patch.object(check_env.metadata, 'distribution', side_effect=check_env.metadata.PackageNotFoundError), \
                patch.object(check_env.shutil, 'which', return_value=None), redirect_stdout(output):
            check_env.main()
        self.assertIn('| PyTorch', output.getvalue())
        self.assertIn('| npu-smi', output.getvalue())
        self.assertIn('| Megatron Core', output.getvalue())
        self.assertIn('| MindSpeed', output.getvalue())
        self.assertIn('| mcore-bridge', output.getvalue())
        self.assertIn('| triton-ascend', output.getvalue())
        self.assertNotIn('| Megatron-LM', output.getvalue())
        sections = [
            '1. Basic environment', '2. Software components', '3. Ascend installation information', '4. NPU status'
        ]
        positions = [output.getvalue().index(section) for section in sections]
        self.assertEqual(positions, sorted(positions))

    def test_found_package_includes_installation_path(self):
        dist = SimpleNamespace(version='2.10.0', locate_file=lambda value: Path('/opt/venv/site-packages'))
        with patch.object(check_env.metadata, 'distribution', return_value=dist):
            version, status, note = check_env.package_version('torch')
        self.assertEqual((version, status), ('2.10.0', 'FOUND'))
        self.assertEqual(note, str(Path('/opt/venv/site-packages')))

    def test_long_installation_path_wraps_without_losing_content(self):
        path = '/' + 'a' * 150
        output = io.StringIO()
        with redirect_stdout(output):
            check_env.print_table([('PyTorch', '2.10.0', 'FOUND', path)])
        lines = output.getvalue().splitlines()
        cells = [line.split('|')[4].strip() for line in lines if line.startswith('|')][1:]
        self.assertEqual(''.join(cells), path)
        self.assertEqual(len({len(line) for line in lines}), 1)

    def test_timeout(self):
        with patch.object(check_env.shutil, 'which', return_value='npu-smi'), \
                patch.object(check_env.subprocess, 'run', side_effect=subprocess.TimeoutExpired('npu-smi', 5)):
            self.assertEqual(check_env.collect(check_env.npu_status)[1], 'UNAVAILABLE')

    def test_empty_device_output_is_unknown(self):
        with patch.object(check_env.shutil, 'which', return_value='npu-smi'), \
                patch.object(check_env.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')):
            self.assertEqual(check_env.npu_status()[1], 'UNKNOWN')

    def test_driver_reports_only_version(self):
        with tempfile.TemporaryDirectory() as directory:
            version = Path(directory) / 'version.info'
            version.write_text('Version=26.0.rc1\nascendhal_version=7.35.23\n', encoding='utf-8')
            self.assertEqual(check_env.read_version([version]), ('26.0.rc1', 'FOUND', str(version.resolve().parent)))

    def test_unrecognized_version_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            version = Path(directory) / 'version.info'
            version.write_text('compatible_version=[8.0,9.0]', encoding='utf-8')
            self.assertEqual(check_env.read_version([version])[1], 'UNKNOWN')

    def test_cann_configured_root(self):
        with tempfile.TemporaryDirectory() as directory:
            version = Path(directory) / 'cann_install.info'
            version.write_text('version=9.1.0', encoding='utf-8')
            with patch.dict(check_env.os.environ, {'ASCEND_HOME_PATH': directory}, clear=True):
                self.assertEqual(check_env.cann_version()[0], '9.1.0')

    def test_permission_error(self):
        with patch.object(Path, 'open', side_effect=PermissionError('denied')):
            self.assertEqual(check_env.read_version(['version.info'])[1], 'UNKNOWN')

    def test_triton_ascend_metadata(self):
        dist = SimpleNamespace(version='3.2.1', locate_file=lambda value: Path('/opt/venv/site-packages'))
        with patch.object(check_env.metadata, 'distribution', return_value=dist) as query:
            self.assertEqual(check_env.package_version('triton-ascend')[:2], ('3.2.1', 'FOUND'))
            query.assert_called_once_with('triton-ascend')


if __name__ == '__main__':
    unittest.main()

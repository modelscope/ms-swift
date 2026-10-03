# Copyright (c) ModelScope Contributors. All rights reserved.
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from swift.cli.main import ROUTE_MAPPING, cli_main


class TestCliMain(unittest.TestCase):

    def assert_help(self, flag):
        result = subprocess.run([sys.executable, '-m', 'swift.cli.main', flag],
                                capture_output=True,
                                text=True,
                                timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Usage: swift <command> [args]', result.stdout)
        commands = result.stdout.split('Available commands: ', 1)[1].splitlines()[0]
        self.assertEqual(commands.split(', '), list(ROUTE_MAPPING))
        self.assertIn('swift <command> --help', result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_long_help(self):
        self.assert_help('--help')

    def test_short_help(self):
        self.assert_help('-h')

    def test_valid_subcommands_route_normally(self):
        for command, module_name in ROUTE_MAPPING.items():
            for args in [['--model', 'test-model'], ['--help'], ['-h']]:
                with self.subTest(command=command, args=args):
                    file_path = f'{command}.py'
                    with patch.object(sys, 'argv', ['swift', command, *args]), \
                            patch('swift.cli.main.importlib.util.find_spec',
                                  return_value=SimpleNamespace(origin=file_path)) as find_spec, \
                            patch('swift.cli.main.get_torchrun_args', return_value=None), \
                            patch('swift.cli.main.subprocess.run',
                                  return_value=SimpleNamespace(returncode=0)) as run, \
                            redirect_stdout(io.StringIO()):
                        self.assertIsNone(cli_main())
                    find_spec.assert_called_once_with(module_name)
                    run.assert_called_once_with([sys.executable, file_path, *args])


if __name__ == '__main__':
    unittest.main()

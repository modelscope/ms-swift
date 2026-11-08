# Copyright (c) ModelScope Contributors. All rights reserved.
import io
import sys
import unittest
from unittest import mock

from swift.cli.main import cli_main


class TestCliMainRouting(unittest.TestCase):
    """`swift --help` (and a bare `swift`) must print the command list instead of raising,
    and an unknown command must fail with a readable message instead of a KeyError."""

    def run_cli(self, argv, route_mapping=None):
        stdout = io.StringIO()
        with mock.patch.object(sys, 'argv', argv), mock.patch('sys.stdout', stdout):
            cli_main(route_mapping=route_mapping)
        return stdout.getvalue()

    def test_help_flags_print_available_commands(self):
        for flag in ('-h', '--help', 'help'):
            with self.subTest(flag=flag):
                out = self.run_cli(['swift', flag])
                self.assertIn('usage', out)
                for command in ('sft', 'rlhf', 'export'):
                    self.assertIn(command, out)

    def test_no_arguments_prints_available_commands(self):
        out = self.run_cli(['swift'])
        self.assertIn('sft', out)

    def test_custom_route_mapping_is_listed(self):
        # the megatron entry point reuses cli_main with its own mapping
        mapping = {'pt': 'swift.cli._megatron.pt', 'sft': 'swift.cli._megatron.sft'}
        out = self.run_cli(['megatron', '--help'], route_mapping=mapping)
        self.assertIn('pt', out)
        self.assertIn('sft', out)

    def test_unknown_command_exits_with_readable_error(self):
        stdout = io.StringIO()
        with mock.patch.object(sys, 'argv', ['swift', 'sftt', '--model', 'x']), \
                mock.patch('sys.stdout', stdout):
            with self.assertRaises(SystemExit) as ctx:
                cli_main()
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn('sftt', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()

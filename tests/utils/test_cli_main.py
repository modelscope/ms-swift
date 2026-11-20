import sys
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

sys.modules['modelscope'] = MagicMock()
sys.modules['modelscope.utils'] = MagicMock()
sys.modules['modelscope.utils.logger'] = MagicMock()

from swift.cli.main import cli_main

class TestCliMain(unittest.TestCase):
    def test_cli_help(self):
        with patch.object(sys, 'argv', ['swift', '--help']), \
             patch('sys.stdout', new_callable=StringIO) as out, \
             self.assertRaises(SystemExit) as cm:
            cli_main()
        self.assertEqual(cm.exception.code, 0)
        output = out.getvalue()
        self.assertIn("Available commands:", output)
        self.assertIn("sft", output)

    def test_cli_no_args(self):
        with patch.object(sys, 'argv', ['swift']), \
             patch('sys.stdout', new_callable=StringIO) as out, \
             self.assertRaises(SystemExit) as cm:
            cli_main()
        self.assertEqual(cm.exception.code, 0)
        output = out.getvalue()
        self.assertIn("Available commands:", output)

    def test_cli_unknown_command(self):
        with patch.object(sys, 'argv', ['swift', 'unknown-cmd']), \
             patch('sys.stdout', new_callable=StringIO) as out, \
             self.assertRaises(SystemExit) as cm:
            cli_main()
        self.assertEqual(cm.exception.code, 1)
        output = out.getvalue()
        self.assertIn("Unknown command: unknown-cmd", output)
        self.assertIn("Available commands:", output)

    @patch('swift.cli.main.subprocess.run')
    def test_cli_valid_command(self, mock_run):
        mock_run.return_value.returncode = 0
        with patch.object(sys, 'argv', ['swift', 'sft', '--help']):
            cli_main()
        self.assertTrue(mock_run.called)
        args = mock_run.call_args[0][0]
        self.assertTrue(any('python' in a.lower() for a in args), f"Expected python executable in {args}")
        self.assertTrue(any('sft.py' in a or 'swift\\cli\\sft' in a or 'swift/cli/sft' in a for a in args), f"Expected sft script in {args}")
        self.assertIn('--help', args)
        self.assertIn('--help', args)

if __name__ == '__main__':
    unittest.main()

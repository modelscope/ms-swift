import os
import subprocess
import sys
import unittest

_SNIPPET = ('from swift.utils.logger import get_logger, ms_logger\n'
            'print(get_logger().level, ms_logger.level)\n')


def _resolve_levels(log_level):
    env = dict(os.environ)
    # swift.utils.logger only applies LOG_LEVEL on the local master, and reads LOCAL_RANK with a
    # bare int(). Drop it so this test measures the level resolution instead of the rank of the
    # process that happens to run it (e.g. inside a torchrun-launched test suite).
    env.pop('LOCAL_RANK', None)
    if log_level is None:
        env.pop('LOG_LEVEL', None)
    else:
        env['LOG_LEVEL'] = log_level
    return subprocess.run([sys.executable, '-c', _SNIPPET], env=env, capture_output=True, text=True, timeout=300)


class TestLogLevelEnvVar(unittest.TestCase):

    def assert_levels(self, log_level, expected):
        proc = _resolve_levels(log_level)
        self.assertEqual(proc.returncode, 0, f'import failed: {proc.stderr}')
        self.assertEqual(proc.stdout.split(), [str(expected), str(expected)])

    def test_unset_uses_info(self):
        self.assert_levels(None, 20)

    def test_valid_level_is_applied(self):
        # a valid level must keep working after the fallback is added
        self.assert_levels('WARNING', 30)

    def test_lowercase_valid_level_is_applied(self):
        self.assert_levels('warning', 30)

    def test_blank_level_does_not_break_import(self):
        # e.g. `LOG_LEVEL=` in a shell script or `ENV LOG_LEVEL=` in a Dockerfile
        self.assert_levels('', 20)

    def test_whitespace_level_falls_back_to_info(self):
        self.assert_levels('  ', 20)

    def test_unknown_level_falls_back_to_info(self):
        self.assert_levels('VERBOSE', 20)


if __name__ == '__main__':
    unittest.main()

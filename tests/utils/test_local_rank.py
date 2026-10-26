import os
import subprocess
import sys
import unittest

_LEVELS_SNIPPET = ('from swift.utils.logger import get_logger, ms_logger\n'
                   'print(get_logger().level, ms_logger.level)\n')

# The second bare LOCAL_RANK read lives in add_file_handler_if_needed, which is only reached once
# the logger is already initialised and has no file handler yet.
_FILE_HANDLER_SNIPPET = ('import os, tempfile\n'
                         'from swift.utils.logger import get_logger\n'
                         'get_logger()\n'
                         'get_logger(log_file=os.path.join(tempfile.mkdtemp(), \'a.log\'))\n'
                         'print(\'HANDLER-OK\')\n')


def _run(snippet, local_rank):
    env = dict(os.environ)
    if local_rank is None:
        env.pop('LOCAL_RANK', None)
    else:
        env['LOCAL_RANK'] = local_rank
    # The level on the local master also depends on LOG_LEVEL, which its own test covers.
    env.pop('LOG_LEVEL', None)
    return subprocess.run([sys.executable, '-c', snippet], env=env, capture_output=True, text=True, timeout=300)


class TestLocalRankEnvVar(unittest.TestCase):

    def assert_levels(self, local_rank, expected):
        proc = _run(_LEVELS_SNIPPET, local_rank)
        self.assertEqual(proc.returncode, 0, f'import failed: {proc.stderr}')
        self.assertEqual(proc.stdout.split(), [str(expected), str(expected)])

    def test_unset_is_treated_as_local_master(self):
        self.assert_levels(None, 20)

    def test_rank_zero_is_treated_as_local_master(self):
        self.assert_levels('0', 20)

    def test_valid_nonzero_rank_still_silences_the_logger(self):
        # a rank that parses must keep its meaning after the fallback is added
        self.assert_levels('1', 40)

    def test_blank_rank_does_not_break_import(self):
        # e.g. `LOCAL_RANK=` in a launcher script or `ENV LOCAL_RANK=` in a Dockerfile
        self.assert_levels('', 20)

    def test_whitespace_rank_falls_back_to_local_master(self):
        self.assert_levels('  ', 20)

    def test_non_numeric_rank_falls_back_to_local_master(self):
        self.assert_levels('placeholder', 20)

    def test_blank_rank_keeps_the_file_handler_path_usable(self):
        proc = _run(_FILE_HANDLER_SNIPPET, '')
        self.assertEqual(proc.returncode, 0, f'get_logger(log_file=...) failed: {proc.stderr}')
        self.assertEqual(proc.stdout.strip(), 'HANDLER-OK')


if __name__ == '__main__':
    unittest.main()

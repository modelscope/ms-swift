import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import swift

# Stands in for `swift/cli/sft.py`. With `handle`, it logs the signal it receives and exits 0 once done,
# taking a second over SIGINT, longer than the 0.25 s that `subprocess.run` waits before killing it.
CHILD = """
import signal
import sys
import time

log, mode = sys.argv[1:]


def handle(signum, frame):
    with open(log, 'a') as f:
        f.write(signal.Signals(signum).name + '\\n')
    if signum == signal.SIGINT:
        time.sleep(1)
    with open(log, 'a') as f:
        f.write('done\\n')
    sys.exit(0)


if mode == 'handle':
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
open(log + '.ready', 'w').close()
time.sleep(120)
"""


@unittest.skipUnless(os.name == 'posix', 'POSIX signals')
class TestCliSignals(unittest.TestCase):
    """`swift sft` runs the trainer (or torchrun) in a child process, so a signal sent to the `swift` process has to
    reach that child instead of leaving it running, or killing it before it can save."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        with open(os.path.join(self._dir.name, 'signal_child.py'), 'w', encoding='utf-8') as f:
            f.write(CHILD)
        self.log = os.path.join(self._dir.name, 'signals.log')

    def launch(self, mode):
        env = {k: v for k, v in os.environ.items() if k not in {'NPROC_PER_NODE', 'NNODES'}}
        swift_root = os.path.dirname(os.path.dirname(os.path.abspath(swift.__file__)))
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [self._dir.name, swift_root, env.get('PYTHONPATH')]))
        code = 'from swift.cli.main import cli_main; cli_main({"child": "signal_child"})'
        # A session of its own, so that the test can signal its process group like a terminal does.
        cli = subprocess.Popen([sys.executable, '-c', code, 'child', self.log, mode],
                               env=env,
                               stdout=subprocess.DEVNULL,
                               start_new_session=True)
        self.addCleanup(self.kill_group, cli)
        deadline = time.monotonic() + 300
        while not os.path.exists(self.log + '.ready'):
            if cli.poll() is not None or time.monotonic() > deadline:
                self.fail(f'the child did not start (CLI exit code {cli.returncode})')
            time.sleep(0.1)
        return cli

    @staticmethod
    def kill_group(cli):
        try:
            os.killpg(cli.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        cli.wait()

    def logged(self):
        with open(self.log, encoding='utf-8') as f:
            return f.read().split()

    def test_sigterm_is_forwarded(self):
        cli = self.launch('handle')
        os.kill(cli.pid, signal.SIGTERM)  # the CLI alone, like Kubernetes or `docker stop`
        self.assertEqual(cli.wait(timeout=60), 0)
        self.assertEqual(self.logged(), ['SIGTERM', 'done'])

    def test_exit_code_of_a_child_killed_by_sigterm(self):
        cli = self.launch('default')
        os.kill(cli.pid, signal.SIGTERM)
        self.assertEqual(cli.wait(timeout=60), 128 + signal.SIGTERM)

    def test_ctrl_c_lets_the_child_finish(self):
        cli = self.launch('handle')
        os.killpg(cli.pid, signal.SIGINT)  # the whole process group, like Ctrl-C in a terminal
        self.assertEqual(cli.wait(timeout=60), 0)
        self.assertEqual(self.logged(), ['SIGINT', 'done'])


if __name__ == '__main__':
    unittest.main()

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Entry scripts that tests launch under ``torchrun`` as fresh subprocesses.

They are not tests themselves -- collecting them would run a training job at import time -- so they
live here under a private name and are addressed by path, never imported.
"""
import os
import signal
import subprocess
from pathlib import Path
from typing import List


class Runners:
    """Resolve a runner script by name, so callers never rebuild the path themselves."""

    DIR = Path(__file__).parent
    #: How a caller gets a free rendezvous port. NOT ``--master_port=0``: that exports
    #: ``MASTER_PORT=0`` to the workers, so rank 0 binds a random port while rank 1 dials port 0 and
    #: the rendezvous hangs until its timeout (measured: a run stuck in ``init_process_group`` with no
    #: error). The c10d backend resolves port 0 to a real free port and publishes it, which also keeps
    #: concurrent pytest processes -- and a lingering rendezvous from the previous case -- apart.
    RENDEZVOUS = ['--rdzv-backend=c10d', '--rdzv-endpoint=localhost:0']

    @staticmethod
    def path(name: str) -> str:
        script = Runners.DIR / f'{name}.py'
        if not script.exists():
            raise FileNotFoundError(f'no runner named {name!r} in {Runners.DIR}')
        return str(script)

    @staticmethod
    def launch(cmd: List[str], *, timeout: int = 1800) -> 'subprocess.CompletedProcess':
        """Run a torchrun command with a timeout that actually fires; return the finished process.

        ``subprocess.run(capture_output=True, timeout=...)`` is not enough here: it kills torchrun
        itself, but the worker grandchildren inherit the pipes, so ``communicate()`` keeps blocking for
        EOF and the timeout never takes effect (one such run hung for 12 hours instead of 30 minutes).
        Giving torchrun its own process group and killing the whole group is what makes it bounded.
        """
        proc = subprocess.Popen(
            cmd,
            env=dict(os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as expired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            out, err = proc.communicate()
            raise AssertionError(f'torchrun exceeded {timeout}s and was killed: {" ".join(cmd)}\n'
                                 f'stdout tail:\n{out[-2000:]}\nstderr tail:\n{err[-3000:]}') from expired
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)

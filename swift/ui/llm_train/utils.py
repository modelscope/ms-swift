# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import os
import re
import subprocess
import sys
from asyncio.subprocess import PIPE, STDOUT
from copy import deepcopy

# Control characters
_CMD_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

# ZIP format signatures:
_CMD_ZIP_SIGNATURES = (b'PK\x03\x04', b'PK\x01\x02', b'PK\x05\x06')


def validate_cmd(cmd: str) -> None:
    """Validate that a command string is safe to write to a shell script.

    Raises:
        ValueError: If forbidden control characters or ZIP signatures are found.
    """
    if _CMD_CONTROL_CHARS.search(cmd):
        raise ValueError('Command contains forbidden control characters')
    cmd_bytes = cmd.encode('utf-8')
    for sig in _CMD_ZIP_SIGNATURES:
        if sig in cmd_bytes:
            raise ValueError('Command contains forbidden ZIP signature bytes')


async def run_and_get_log(*args, timeout=None):
    process = await asyncio.create_subprocess_exec(*args, stdout=PIPE, stderr=STDOUT)
    lines = []
    while True:
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            break
        else:
            if not line:
                break
            else:
                lines.append(str(line))
    return process, lines


def run_command_in_subprocess(*args, timeout):
    if sys.platform == 'win32':
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    process, lines = loop.run_until_complete(run_and_get_log(*args, timeout=timeout))
    return (loop, process), lines


def close_loop(handler):
    loop, process = handler
    process.kill()
    loop.close()


def run_command_in_background_with_popen(command, all_envs, log_file):
    env = deepcopy(os.environ)
    if len(all_envs) > 0:
        for k, v in all_envs.items():
            env[k] = v
    daemon_kwargs = {}
    if sys.platform == 'win32':
        from subprocess import CREATE_NO_WINDOW, DETACHED_PROCESS
        daemon_kwargs['creationflags'] = DETACHED_PROCESS | CREATE_NO_WINDOW
        daemon_kwargs['close_fds'] = True
    else:
        daemon_kwargs['preexec_fn'] = os.setsid

    with open(log_file, 'w', encoding='utf-8') as f:
        subprocess.Popen(
            command, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1, env=env)

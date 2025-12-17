# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
import subprocess
import sys
from asyncio.subprocess import PIPE, STDOUT
from copy import deepcopy


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
        from subprocess import DETACHED_PROCESS, CREATE_NO_WINDOW
        daemon_kwargs['creationflags'] = DETACHED_PROCESS | CREATE_NO_WINDOW
        daemon_kwargs['close_fds'] = True
    else:
        daemon_kwargs['preexec_fn'] = os.setsid

    with open(log_file, 'w', encoding='utf-8') as f:
        subprocess.Popen(
            command, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, bufsize=1, env=env)

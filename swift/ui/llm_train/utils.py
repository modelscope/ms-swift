# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import sys
from asyncio.subprocess import PIPE, STDOUT


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

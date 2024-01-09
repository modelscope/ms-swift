import asyncio
import sys
from asyncio.subprocess import PIPE, STDOUT


async def run_and_get_log(*args, timeout=None):
    process = await asyncio.create_subprocess_exec(
        *args, stdout=PIPE, stderr=STDOUT)
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
    process, lines = loop.run_until_complete(
        run_and_get_log(*args, timeout=timeout))
    return (loop, process), lines


class TailContext:

    def __init__(self):
        self.loop = None
        self.process = None

    def __enter__(self):
        if sys.platform == 'win32':
            self.loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(self.loop)
        else:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self

    async def run_and_yield_log(self, *args, timeout=None):
        self.process = await asyncio.create_subprocess_exec(
            *args, stdout=PIPE, stderr=STDOUT)
        while True:
            try:
                line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
            except asyncio.TimeoutError:
                break
            else:
                if not line:
                    break
                else:
                    async yield line

    def tail(self, filename, timeout):
        self.loop.run_until_complete(self.run_and_yield_log(['tail', '-F', filename], timeout=timeout))

    def __exit__(self, type, value, traceback):
        self.process.kill()
        self.loop.close()



def close_loop(handler):
    loop, process = handler
    process.kill()
    loop.close()

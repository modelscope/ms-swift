# Copyright (c) ModelScope Contributors. All rights reserved.
"""Print a compact Ascend environment report without importing training packages."""
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from importlib import metadata
from pathlib import Path


def package_version(name):
    try:
        dist = metadata.distribution(name)
        return dist.version, 'FOUND', str(dist.locate_file(''))
    except metadata.PackageNotFoundError:
        return '-', 'MISSING', 'Package metadata not found'


def read_version(candidates):
    """Read only explicit version fields, never infer versions from path names."""
    seen = set()
    unreadable = False
    for candidate in candidates:
        path = Path(candidate).expanduser()
        try:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            with path.open(encoding='utf-8', errors='replace') as handle:
                content = handle.read(8192)
            for key in ('version', 'package_version', 'cann_version'):
                match = re.search(rf'^\s*{key}\s*[:=]\s*[\"\']?([^\s\"\']+)', content, re.MULTILINE | re.IGNORECASE)
                if match:
                    return match.group(1), 'FOUND', str(resolved.parent)
        except FileNotFoundError:
            continue
        except OSError:
            unreadable = True
    note = 'Version file could not be read' if unreadable else 'Version not detected; installation may still exist'
    return '-', 'UNKNOWN', note


def driver_version():
    return read_version(['/usr/local/Ascend/driver/version.info'])


def cann_version():
    configured = [os.environ[key] for key in ('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME') if os.environ.get(key)]
    # Explicit roots take precedence; do not report a different standard installation's version.
    roots = configured or ['/usr/local/Ascend/ascend-toolkit/latest', '~/Ascend/ascend-toolkit/latest']
    candidates = []
    for root in roots:
        path = Path(root).expanduser()
        for subdir in ('', 'aarch64-linux', 'x86_64-linux'):
            for filename in ('version.cfg', 'version.info', 'ascend_toolkit_install.info', 'cann_install.info'):
                candidates.append(path / subdir / filename)
    version, status, note = read_version(candidates)
    if status != 'FOUND' and configured:
        note += f'; configured path: {configured[0]}'
    return version, status, note


def npu_status():
    executable = shutil.which('npu-smi')
    if executable is None:
        return '-', 'MISSING', 'npu-smi not found on PATH', ''
    result = subprocess.run([executable, 'info'],
                            capture_output=True,
                            text=True,
                            errors='replace',
                            timeout=5,
                            check=False)
    if result.returncode:
        reason = ' '.join((result.stderr or result.stdout).split())
        return '-', 'UNAVAILABLE', f'exit={result.returncode}: {reason}'[:80], ''
    if not result.stdout.strip():
        return '-', 'UNKNOWN', 'npu-smi returned no device information', ''
    return '-', 'FOUND', executable, result.stdout.strip()


def collect(check, *args):
    try:
        return check(*args)
    except subprocess.TimeoutExpired:
        return '-', 'UNAVAILABLE', 'Query timed out after 5 seconds'
    except Exception as exc:
        # Keep reporting other components when a single probe fails.
        return '-', 'UNAVAILABLE', f'{type(exc).__name__}: {" ".join(str(exc).split())}'[:80]


def print_table(rows):
    rows = [('Component', 'Version', 'Status', 'Note'), *rows]
    # Wrap long installation paths instead of widening the entire table.
    limits = (24, 48, 12, 72)
    wrapped = [[
        textwrap.wrap(str(value), width=limits[i], break_on_hyphens=False) or [''] for i, value in enumerate(row)
    ] for row in rows]
    widths = [max(len(line) for row in wrapped for line in row[i]) for i in range(4)]
    separator = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'
    print(separator)
    for index, row in enumerate(wrapped):
        for line_index in range(max(len(cell) for cell in row)):
            cells = [cell[line_index] if line_index < len(cell) else '' for cell in row]
            print('| ' + ' | '.join(value.ljust(widths[i]) for i, value in enumerate(cells)) + ' |')
        if index == 0:
            print(separator)
    print(separator)


def main():
    print('Ascend environment')
    print('\n1. Basic environment')
    print_table([
        ('OS', f'{platform.system()} {platform.release()}', 'FOUND', platform.machine()),
        ('Python', platform.python_version(), 'FOUND', sys.executable),
    ])
    print('\n2. Software components')
    packages = (
        ('PyTorch', 'torch'),
        ('torch-npu', 'torch-npu'),
        ('ms-swift', 'ms-swift'),
        ('Megatron Core', 'megatron-core'),
        ('MindSpeed', 'mindspeed'),
        ('mcore-bridge', 'mcore-bridge'),
        ('triton-ascend', 'triton-ascend'),
    )
    print_table([(label, *collect(package_version, package)) for label, package in packages])
    print('\n3. Ascend installation information')
    print_table([('Ascend driver', *collect(driver_version)), ('CANN', *collect(cann_version))])
    print('\n4. NPU status')
    npu = collect(npu_status)
    print_table([('npu-smi', *npu[:3])])
    if len(npu) == 4 and npu[3]:
        print('\nNPU status (npu-smi):')
        print(npu[3])
    print('\nFOUND means information was detected; compatibility and runtime availability are not checked.')
    print('Missing backend-specific components does not necessarily indicate an unusable training environment.')


if __name__ == '__main__':
    main()

# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16 L1 static consistency checks over the examples/ tree.

The v5 examples carry their routing contract in the file itself: ``USE_SWIFT_V5=1`` selects the
dev CLI, and the entry the script then calls must be one of the subcommands that actually HAS a
dev route (``swift/cli/main.py DEV_ROUTE_MAPPING``). A script that sets the flag and calls
``swift rollout`` (no route: ``resolve_route`` raises "no dev route exists") is a broken example
-- it dies at argv parsing, before any training logic. These tests scan the tree statically so
that class of rot is caught at collection time, not when a user runs the script.

Also pinned here: the retired entries (``sample`` / ``merge-lora``, redirected by
``DEV_ROUTE_REDIRECTS``; ``rollout``, legacy-only) never appear in a v5-flagged script, and the
NPU (ascend) group self-identifies so a CUDA-side collection can skip it cleanly instead of
mis-running it.
"""
import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[5] / 'examples'

#: The set of subcommands a USE_SWIFT_V5=1 script may invoke. Read from the router itself so this
#: test cannot drift from the real dispatch table.
def _dev_routes():
    from swift.cli.main import DEV_ROUTE_MAPPING
    return DEV_ROUTE_MAPPING


def _sh_files():
    return sorted(EXAMPLES.rglob('*.sh'))


def _is_comment(line):
    return line.lstrip().startswith('#')


def _invoked_subcommands(text):
    """Every `swift <sub>` the script actually invokes, ignoring comment lines."""
    subs = []
    for line in text.splitlines():
        if _is_comment(line):
            continue
        match = re.match(r'\s*swift\s+([a-z][a-z0-9_-]*)', line)
        if match:
            subs.append(match.group(1))
    return subs


def _logical_lines(text):
    """Comment-stripped lines with ``\\``-continuations joined back into one logical line."""
    lines = [line for line in text.splitlines() if not _is_comment(line)]
    joined, buf = [], ''
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith('\\'):
            buf += stripped[:-1] + ' '
        else:
            joined.append((buf + stripped).strip())
            buf = ''
    if buf:
        joined.append(buf.strip())
    return joined


def _swift_invocations(text):
    """``(subcommand, argv)`` for every ``swift <sub> ...`` logical line in the script."""
    out = []
    for line in _logical_lines(text):
        tokens = line.split()
        for i, token in enumerate(tokens):
            if token == 'swift' and i + 1 < len(tokens):
                out.append((tokens[i + 1], tokens[i + 2:]))
                break
    return out


def test_v5_examples_only_call_routed_subcommands():
    """A USE_SWIFT_V5=1 script must call only subcommands with a dev route -- anything else dies
    at resolve_route before the first training step."""
    routes = _dev_routes()
    offenders = []
    for sh in _sh_files():
        text = sh.read_text(encoding='utf-8', errors='replace')
        if 'USE_SWIFT_V5=1' not in text:
            continue
        for sub in _invoked_subcommands(text):
            if sub not in routes:
                offenders.append(f'{sh.relative_to(EXAMPLES)}: swift {sub}')
    assert not offenders, (
        'v5-flagged examples invoke subcommands with no dev route (they would raise at '
        f'resolve_route): {offenders}')


def test_retired_and_legacy_only_entries_never_riding_the_v5_flag():
    """`sample` / `merge-lora` are redirected (v5 spellings exist) and `rollout` / `app` have no
    dev route at all; a script using them must NOT also set USE_SWIFT_V5=1 -- the flag would make
    resolve_route raise instead of running the legacy path the script was written for."""
    offenders = []
    for sh in _sh_files():
        text = sh.read_text(encoding='utf-8', errors='replace')
        if 'USE_SWIFT_V5=1' not in text:
            continue
        for sub in _invoked_subcommands(text):
            if sub in ('sample', 'merge-lora', 'rollout', 'app', 'web-ui'):
                offenders.append(f'{sh.relative_to(EXAMPLES)}: swift {sub}')
    assert not offenders, f'retired/legacy-only entries combined with USE_SWIFT_V5=1: {offenders}'


def test_ascend_examples_self_identify_and_stay_off_the_v5_flag():
    """S16-20: every script under examples/ascend carries a machine-readable NPU marker
    (ASCEND_RT_VISIBLE_DEVICES / PYTORCH_NPU_ALLOC_* / any npu spelling) so a CUDA-side collection
    can skip the group cleanly, and none of them sets USE_SWIFT_V5=1 (the NPU group is not v5-
    migrated; the flag would route them to the dev CLI that has not been validated on NPU)."""
    ascend = EXAMPLES / 'ascend'
    scripts = sorted(ascend.rglob('*.sh'))
    assert scripts, 'examples/ascend disappeared; update this test with the tree'
    unmarked, flagged = [], []
    for sh in scripts:
        text = sh.read_text(encoding='utf-8', errors='replace')
        if not re.search(r'ASCEND_RT_VISIBLE_DEVICES|PYTORCH_NPU_ALLOC|[Nn][Pp][Uu]', text):
            unmarked.append(str(sh.relative_to(EXAMPLES)))
        if 'USE_SWIFT_V5=1' in text:
            flagged.append(str(sh.relative_to(EXAMPLES)))
    assert not unmarked, f'ascend scripts with no NPU marker (cannot be skipped cleanly): {unmarked}'
    assert not flagged, f'ascend scripts setting USE_SWIFT_V5=1 (NPU group is not v5-migrated): {flagged}'


def test_v5_examples_declare_their_routing_in_the_header():
    """Every v5-flagged script's header explains the flag (mentions v5/dev routing), so a reader
    of the file knows the entry is the dev CLI, not legacy swift."""
    silent = []
    for sh in _sh_files():
        text = sh.read_text(encoding='utf-8', errors='replace')
        if 'USE_SWIFT_V5=1' not in text:
            continue
        header = '\n'.join(line for line in text.splitlines()[:30] if _is_comment(line))
        if not re.search(r'USE_SWIFT_V5|swift\.dev|dev CLI|v5', header):
            silent.append(str(sh.relative_to(EXAMPLES)))
    assert not silent, f'v5-flagged scripts whose header never mentions the routing: {silent}'


def test_v5_examples_survive_the_legacy_flag_gate():
    """Every USE_SWIFT_V5=1 script must pass the real retired-flag gate ``reject_legacy_only_flags``
    -- a v5-flagged example carrying an obsolete option dies at argv parsing, before any training
    logic, with a refusal the script author never intended. The gate is the product's own
    classification (CLI_LEGACY_ONLY per subcommand, megatron-suffixed when ``--backend megatron``),
    so this test cannot drift from the actual rejection behavior."""
    from swift.dev.cli._megatron_compat import is_megatron_argv
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags

    failures = []
    for sh in _sh_files():
        text = sh.read_text(encoding='utf-8', errors='replace')
        if 'USE_SWIFT_V5=1' not in text:
            continue
        for sub, argv in _swift_invocations(text):
            command = f'megatron_{sub}' if is_megatron_argv(argv) else sub
            try:
                reject_legacy_only_flags(command, argv)
            except ValueError as exc:
                failures.append(f'{sh.relative_to(EXAMPLES)}: {exc}')
    assert not failures, (
        'v5-flagged examples use options the dev CLI refuses at parse time (they die before the '
        f'first training step): {failures}')

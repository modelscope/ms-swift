#!/usr/bin/env python3
"""Generate placeholder-only AgentArk datasets for ms-swift GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_RUN_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]*$')
_MAX_GROUP_SEED = 2**31 - 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=('Generate unique AgentArk sampling tickets. Each JSONL row represents '
                     'one GRPO prompt group; ms-swift repeats it num_generations times.'))
    parser.add_argument('--output', type=Path, required=True, help='Output JSONL path')
    parser.add_argument('--run-id', required=True, help='Unique experiment/run prefix')
    parser.add_argument('--count', type=int, required=True, help='Number of unique group tickets')
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='First numeric ticket index (useful when extending a run)',
    )
    parser.add_argument(
        '--task-name',
        default=None,
        help='Optionally pin every ticket to one AgentArk task',
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        '--group-seed',
        type=int,
        default=None,
        help='Optionally pin every ticket to the same group seed',
    )
    seed_group.add_argument(
        '--group-seed-base',
        type=int,
        default=None,
        help='Assign group_seed = base + ticket index',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Replace an existing output file atomically',
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not _RUN_ID_RE.fullmatch(args.run_id):
        raise SystemExit('--run-id must start with an alphanumeric character and contain only '
                         "letters, digits, '.', '_', ':', or '-'")
    if args.count <= 0:
        raise SystemExit('--count must be greater than zero')
    if args.start_index < 0:
        raise SystemExit('--start-index must be non-negative')
    if args.output.exists() and not args.force:
        raise SystemExit(f'Output already exists (pass --force to replace it): {args.output}')
    if args.task_name is not None and not args.task_name.strip():
        raise SystemExit('--task-name cannot be empty')
    for option, value in (
        ('--group-seed', args.group_seed),
        ('--group-seed-base', args.group_seed_base),
    ):
        if value is not None and not 1 <= value <= _MAX_GROUP_SEED:
            raise SystemExit(f'{option} must be in [1, {_MAX_GROUP_SEED}]')
    if args.group_seed_base is not None:
        final_seed = args.group_seed_base + args.start_index + args.count - 1
        if final_seed > _MAX_GROUP_SEED:
            raise SystemExit(f'Sequential group seeds end at {final_seed}, above the maximum {_MAX_GROUP_SEED}')


def _stable_group_seed(group_uid: str) -> int:
    digest = hashlib.sha256(group_uid.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % _MAX_GROUP_SEED + 1


def _ticket(args: argparse.Namespace, index: int) -> dict[str, Any]:
    group_uid = f'{args.run_id}:{index:08d}'
    env_config: dict[str, Any] = {
        'name': 'agentark',
        'group_uid': group_uid,
    }
    if args.task_name is not None:
        env_config['task_name'] = args.task_name.strip()
    if args.group_seed is not None:
        env_config['group_seed'] = args.group_seed
    elif args.group_seed_base is not None:
        env_config['group_seed'] = args.group_seed_base + index
    elif args.task_name is not None:
        # When task_name is pinned, the AgentArk server intentionally skips its
        # uid -> (task, seed) selector. Derive a seed here so G sibling
        # trajectories still reset to the same task state.
        env_config['group_seed'] = _stable_group_seed(group_uid)

    return {
        'messages': [{
            'role': 'user',
            'content': f'<agentark-ticket:{group_uid}>',
        }],
        'env_config': env_config,
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=args.output.parent,
                prefix=f'.{args.output.name}.',
                suffix='.tmp',
                delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for offset in range(args.count):
                index = args.start_index + offset
                json.dump(_ticket(args, index), handle, ensure_ascii=False, separators=(',', ':'))
                handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, args.output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    final_index = args.start_index + args.count - 1
    seed_mode = 'server-managed'
    if args.group_seed is not None:
        seed_mode = f'fixed:{args.group_seed}'
    elif args.group_seed_base is not None:
        seed_mode = f'sequential:{args.group_seed_base + args.start_index}..{args.group_seed_base + final_index}'
    elif args.task_name is not None:
        seed_mode = 'stable-derived-per-group'
    print(
        json.dumps(
            {
                'output': str(args.output),
                'rows': args.count,
                'first_group_uid': f'{args.run_id}:{args.start_index:08d}',
                'last_group_uid': f'{args.run_id}:{final_index:08d}',
                'task': args.task_name or 'server-managed',
                'seed': seed_mode,
            },
            ensure_ascii=False,
        ))


if __name__ == '__main__':
    main()

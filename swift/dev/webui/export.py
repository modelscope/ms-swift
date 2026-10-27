# Copyright (c) ModelScope Contributors. All rights reserved.
"""Export the registry and channel schemas as static JSON for the frontend.

Usage: ``python -m swift.dev.webui.export [out.json]`` (stdout when omitted).
The frontend treats the dumped file as build input; the Python registry stays
the single source of truth.
"""
from __future__ import annotations

import json
import sys

from .registry import REGISTRY, validate_registry
from .schema import CHANNEL_SCHEMAS


def build_payload() -> dict:
    """The full declarative payload: components + channel contracts + roles."""
    validate_registry()
    return {
        'channels': {kind: schema.to_dict() for kind, schema in CHANNEL_SCHEMAS.items()},
        'components': {d.type: d.to_dict() for d in REGISTRY.values()},
        'version': 1,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    payload = build_payload()
    if not argv:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    else:
        with open(argv[0], 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'wrote {argv[0]}: {len(payload["components"])} components, '
              f'{len(payload["channels"])} channels')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

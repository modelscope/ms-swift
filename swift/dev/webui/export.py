# Copyright (c) ModelScope Contributors. All rights reserved.
"""Export the registry and channel schemas as static JSON for the frontend.

Usage: ``python -m swift.dev.webui.export [out.json]`` (stdout when omitted).
The frontend treats the dumped file as build input; the Python registry stays
the single source of truth.

The old config-consumability check (every registry param had to land on a dev
Config field via the recipe-translation rename tables) is gone with the recipe
executor: params now bind to REAL component constructors and methods, and the
binding-time check is the component's own loud failure (``_fill`` rejects
unknown names). What survives here is the structural check: every port kind
and profile must exist, and every declared factory path must be importable.
"""
from __future__ import annotations

import importlib
import json
import sys

from .registry import REGISTRY, REGISTRY_VERSION, validate_registry
from .schema import CHANNEL_SCHEMAS


def validate_factories() -> None:
    """Every declared factory path must resolve to a real importable object.

    ``module:attr`` (and ``module`` alone -- the constructor is the module's
    namesake, e.g. swift.dev.builders.dataset) both resolve here. A stale
    factory is caught at export time, not at first canvas click.
    """
    for descriptor in REGISTRY.values():
        path = descriptor.factory or descriptor.client_factory
        if not path:
            continue
        module_path, _, attr = path.partition(':')
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(f'{descriptor.type}: factory module {module_path!r} '
                             f'does not import: {exc}') from None
        if attr and not hasattr(module, attr):
            raise ValueError(f'{descriptor.type}: factory {path!r} resolves to nothing')


def build_payload() -> dict:
    """The full declarative payload: components + channel contracts + laws."""
    validate_registry()
    return {
        'channels': {kind: schema.to_dict() for kind, schema in CHANNEL_SCHEMAS.items()},
        'components': {d.type: d.to_dict() for d in REGISTRY.values()},
        'version': REGISTRY_VERSION,
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

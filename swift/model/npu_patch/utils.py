# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()


def import_optional_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        logger.debug('Failed to import optional module %s: %s', module_name, exc)
        return None


def setattr_path(root: Any, path: str, value: Any) -> None:
    current = root
    parts = path.split('.')
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def apply_patch_map(root: Any, patch_map: dict[str, Any]) -> None:
    for path, value in patch_map.items():
        setattr_path(root, path, value)

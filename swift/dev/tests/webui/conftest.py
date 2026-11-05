# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pytest config for the webui orchestration tests.

Two environment duties the parent conftest does not carry:

* ``twinkle`` is a nested source checkout (``twinkle/src``), not an installed
  package on every box: prepend it to ``sys.path`` when the import cannot
  resolve, so the component bindings (``twinkle.loss`` and friends) import.
* preset-contract cases drive the REAL frontend ``presets.ts`` through
  esbuild; the ``requires_node`` marker skips them where the webui toolchain
  is absent instead of erroring mid-run.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

#: The nested twinkle checkout this repo carries (sibling of swift/).
_TWINKLE_SRC = Path(__file__).resolve().parents[4] / 'twinkle' / 'src'

#: The webui frontend package whose node_modules holds esbuild.
_WEBUI_DIR = Path(__file__).resolve().parents[4] / 'web-ui-v5' / 'webui'


def _twinkle_resolves() -> bool:
    try:
        return importlib.util.find_spec('twinkle.loss') is not None
    except (ImportError, ValueError):
        return False


if not _twinkle_resolves():
    sys.path.insert(0, str(_TWINKLE_SRC))
    # ``find_spec`` above imports the parent and the repo root's ``twinkle/``
    # directory (the nested checkout root, no top-level ``__init__.py``)
    # resolves as a NAMESPACE package, which lands in ``sys.modules`` and then
    # shadows the real package under ``twinkle/src`` for every later import.
    # Evict that stub so the insertion actually takes effect.
    _stale = sys.modules.get('twinkle')
    if _stale is not None and getattr(_stale, '__file__', None) is None:
        del sys.modules['twinkle']

    # Ray workers import twinkle by PYTHONPATH, not by the driver's sys.path;
    # keep the environment consistent with what this process just resolved.
    _repo_root = str(_TWINKLE_SRC.parent.parent)
    _existing = os.environ.get('PYTHONPATH', '')
    _parts = [part for part in _existing.split(os.pathsep) if part]
    for _entry in (str(_TWINKLE_SRC), _repo_root):
        if _entry not in _parts:
            _parts.insert(0, _entry)
    os.environ['PYTHONPATH'] = os.pathsep.join(_parts)


def _esbuild_available() -> bool:
    return (_WEBUI_DIR / 'node_modules' / '.bin' / 'esbuild').exists()


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'requires_node: needs the webui node toolchain (skip when absent)')


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker('requires_node') is not None and not _esbuild_available():
            item.add_marker(pytest.mark.skip(reason='webui node_modules/esbuild not installed'))

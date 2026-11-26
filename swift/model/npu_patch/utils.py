# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib
import torch
from typing import Any

from swift.utils.logger import get_logger

logger = get_logger()


def import_optional_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        logger.debug('Failed to import optional module %s: %s', module_name, exc)
        return None


def apply_patch_map(root: Any, patch_map: dict[str, Any]) -> None:
    for path, value in patch_map.items():
        current = root
        parts = path.split('.')
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], value)


def get_cann_mhc_ops():
    try:
        from cann_ops_transformer.ops import mhc_post, mhc_pre_sinkhorn
    except (ImportError, OSError, RuntimeError) as exc:
        logger.debug('CANN mHC unavailable: %s', exc)
        return None
    return mhc_pre_sinkhorn, mhc_post


def supports_cann_mhc(x, streams, width):
    return (x.device.type == 'npu' and x.dtype == torch.bfloat16 and x.numel() > 0 and streams == 4
            and width in (1280, 1920, 2560, 4096))

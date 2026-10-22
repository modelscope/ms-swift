"""Small runtime helpers used by dev templates.

This module deliberately owns the few generic utilities templates need, so family modules do not
import the legacy ``swift.utils`` package.
"""

from __future__ import annotations
import logging
import os
import time
from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import torch
from transformers.utils import strtobool

_T = TypeVar("_T")


class _OnceLogger(logging.LoggerAdapter):
    def __init__(self, logger):
        super().__init__(logger, {})
        self._seen = set()

    def _once(self, level: int, message: str, *args, hash_id=None, **kwargs):
        key = hash_id or (level, message)
        if key in self._seen:
            return
        self._seen.add(key)
        self.log(level, message, *args, **kwargs)

    def info_once(self, message: str, *args, **kwargs):
        self._once(logging.INFO, message, *args, **kwargs)

    def warning_once(self, message: str, *args, **kwargs):
        self._once(logging.WARNING, message, *args, **kwargs)


_LOGGER = _OnceLogger(logging.getLogger("swift.dev.template"))


def get_logger():
    return _LOGGER


def get_env_args(name: str, convert: Callable[[str], _T], default: Optional[_T]) -> Optional[_T]:
    value = os.getenv(name.upper())
    if value is None:
        return default
    if convert is bool:  # bool('false') is True: parse the usual 0/1/yes/no/on/off words instead
        return bool(strtobool(value))
    return convert(value)


def upper_bound(lo: int, hi: int, condition: Callable[[int], bool]) -> int:
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if condition(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def retry_decorator(retry: int = 3):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            last = None
            for attempt in range(retry):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    last = error
                    if attempt + 1 < retry:
                        time.sleep(0.1)
            raise last

        return wrapped

    return decorate


def remove_response(messages) -> Optional[str]:
    if messages and messages[-1].get("role") == "assistant":
        return messages.pop()["content"]
    return None


def to_device(value: Any, device, non_blocking: bool = False):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return type(value)((key, to_device(item, device, non_blocking)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [to_device(item, device, non_blocking) for item in value]
    return value


def to_float_dtype(value: Any, dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, Mapping):
        return type(value)((key, to_float_dtype(item, dtype)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(to_float_dtype(item, dtype) for item in value)
    if isinstance(value, list):
        return [to_float_dtype(item, dtype) for item in value]
    return value


def is_deepspeed_zero3_enabled() -> bool:
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled

        return bool(is_deepspeed_zero3_enabled())
    except Exception:
        return False


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

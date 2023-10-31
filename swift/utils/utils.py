# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import re
from typing import (Any, Callable, List, Mapping, Optional, Sequence, Tuple,
                    Type, TypeVar)

from transformers import HfArgumentParser

from .logger import get_logger

logger = get_logger()


def check_json_format(obj: Any) -> Any:
    if obj is None or isinstance(
            obj, (int, float, str, complex)):  # bool is a subclass of int
        return obj

    if isinstance(obj, Sequence):
        res = []
        for x in obj:
            res.append(check_json_format(x))
    elif isinstance(obj, Mapping):
        res = {}
        for k, v in obj.items():
            res[k] = check_json_format(v)
    else:
        res = repr(obj)  # e.g. function
    return res


def _get_version(work_dir: str) -> int:
    if os.path.isdir(work_dir):
        fnames = os.listdir(work_dir)
    else:
        fnames = []
    v_list = [-1]
    for fname in fnames:
        m = re.match(r'v(\d+)', fname)
        if m is None:
            continue
        v = m.group(1)
        v_list.append(int(v))
    return max(v_list) + 1


def add_version_to_work_dir(work_dir: str) -> str:
    """add version"""
    version = _get_version(work_dir)
    time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')

    work_dir = os.path.join(work_dir, f'v{version}-{time}')
    logger.info(f'work_dir: {work_dir}')
    return work_dir


_T = TypeVar('_T')


def parse_args(class_type: Type[_T],
               argv: Optional[List[str]] = None) -> Tuple[_T, List[str]]:
    parser = HfArgumentParser([class_type])
    args, remaining_args = parser.parse_args_into_dataclasses(
        argv, return_remaining_strings=True)
    return args, remaining_args


def lower_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The lower bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi) >> 1
        if cond(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def upper_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The upper bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi + 1) >> 1  # lo + (hi-lo+1)>>1
        if cond(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo

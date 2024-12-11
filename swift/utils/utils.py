# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import random
import re
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from transformers import HfArgumentParser, enable_full_determinism, set_seed

from .env import is_dist, is_dist_ta
from .logger import get_logger
from .np_utils import stat_array

logger = get_logger()


def check_json_format(obj: Any, token_safe: bool = True) -> Any:
    if obj is None or isinstance(obj, (int, float, str, complex)):  # bool is a subclass of int
        return obj
    if isinstance(obj, bytes):
        return '<<<bytes>>>'
    if isinstance(obj, (torch.dtype, torch.device)):
        obj = str(obj)
        return obj[len('torch.'):] if obj.startswith('torch.') else obj

    if isinstance(obj, Sequence):
        res = []
        for x in obj:
            res.append(check_json_format(x, token_safe))
    elif isinstance(obj, Mapping):
        res = {}
        for k, v in obj.items():
            if token_safe and isinstance(k, str) and '_token' in k:
                res[k] = None
            else:
                res[k] = check_json_format(v, token_safe)
    else:
        if token_safe:
            unsafe_items = {}
            for k, v in obj.__dict__.items():
                if '_token' in k:
                    unsafe_items[k] = v
                    setattr(obj, k, None)
            res = repr(obj)
            # recover
            for k, v in unsafe_items.items():
                setattr(obj, k, v)
        else:
            res = repr(obj)  # e.g. function, object
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


def format_time(seconds):
    days = int(seconds // (24 * 3600))
    hours = int((seconds % (24 * 3600)) // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if days > 0:
        time_str = f'{days}d {hours}h {minutes}m {seconds}s'
    elif hours > 0:
        time_str = f'{hours}h {minutes}m {seconds}s'
    elif minutes > 0:
        time_str = f'{minutes}m {seconds}s'
    else:
        time_str = f'{seconds}s'

    return time_str


def deep_getattr(obj, attr: str, default=None):
    attrs = attr.split('.')
    for a in attrs:
        if obj is None:
            break
        if isinstance(obj, dict):
            obj = obj.get(a, default)
        else:
            obj = getattr(obj, a, default)
    return obj


def seed_everything(seed: Optional[int] = None, full_determinism: bool = False, *, verbose: bool = True) -> int:

    if seed is None:
        seed_max = np.iinfo(np.int32).max
        seed = random.randint(0, seed_max)

    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed(seed)
    if verbose:
        logger.info(f'Global seed set to {seed}')
    return seed


def add_version_to_work_dir(work_dir: str) -> str:
    """add version"""
    version = _get_version(work_dir)
    time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    sub_folder = f'v{version}-{time}'
    if (dist.is_initialized() and is_dist()) or is_dist_ta():
        obj_list = [sub_folder]
        dist.broadcast_object_list(obj_list)
        sub_folder = obj_list[0]

    work_dir = os.path.join(work_dir, sub_folder)
    return work_dir


_T = TypeVar('_T')


def parse_args(class_type: Type[_T], argv: Optional[List[str]] = None) -> Tuple[_T, List[str]]:
    parser = HfArgumentParser([class_type])
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) > 0 and argv[0].endswith('.json'):
        json_path = os.path.abspath(os.path.expanduser(argv[0]))
        args, = parser.parse_json_file(json_path)
        remaining_args = argv[1:]
    else:
        args, remaining_args = parser.parse_args_into_dataclasses(argv, return_remaining_strings=True)
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


def test_time(func: Callable[[], _T],
              number: int = 1,
              warmup: int = 0,
              timer: Optional[Callable[[], float]] = None) -> _T:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter

    ts = []
    res = None
    # warmup
    for _ in range(warmup):
        res = func()

    for _ in range(number):
        t1 = timer()
        res = func()
        t2 = timer()
        ts.append(t2 - t1)

    ts = np.array(ts)
    _, stat_str = stat_array(ts)
    # print
    logger.info(f'time[number={number}]: {stat_str}')
    return res


def read_multi_line(addi_prompt: str = '') -> str:
    res = []
    prompt = f'<<<{addi_prompt} '
    while True:
        text = input(prompt) + '\n'
        prompt = ''
        res.append(text)
        if text.endswith('#\n'):
            res[-1] = text[:-2]
            break
    return ''.join(res)


def subprocess_run(command: List[str], env: Optional[Dict[str, str]] = None, stdout=None, stderr=None):
    # stdoutm stderr: e.g. subprocess.PIPE.
    resp = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    resp.check_returncode()
    return resp


def get_env_args(args_name: str, type_func: Callable[[str], _T], default_value: Optional[_T]) -> Optional[_T]:
    args_name_upper = args_name.upper()
    value = os.getenv(args_name_upper)
    if value is None:
        value = default_value
        log_info = (f'Setting {args_name}: {default_value}. '
                    f'You can adjust this hyperparameter through the environment variable: `{args_name_upper}`.')
    else:
        value = type_func(value)
        log_info = f'Using environment variable `{args_name_upper}`, Setting {args_name}: {value}.'
    logger.info_once(log_info)
    return value


def find_free_port(start_port: Optional[int] = None, retry: int = 100) -> int:
    if start_port is None:
        start_port = 0
    for port in range(start_port, start_port + retry):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('', start_port))
                port = sock.getsockname()[1]
                break
            except OSError:
                pass
    return port


def split_list(ori_list, num_shards):
    idx_list = np.linspace(0, len(ori_list), num_shards + 1)
    shard = []
    for i in range(len(idx_list) - 1):
        shard.append(ori_list[int(idx_list[i]):int(idx_list[i + 1])])
    return shard


def patch_getattr(obj_cls, item_name: str):
    if hasattr(obj_cls, '_patch'):  # avoid double patch
        return

    def __new_getattr__(self, key: str):
        try:
            return super(self.__class__, self).__getattr__(key)
        except AttributeError:
            if item_name in dir(self):
                item = getattr(self, item_name)
                return getattr(item, key)
            raise

    obj_cls.__getattr__ = __new_getattr__
    obj_cls._patch = True

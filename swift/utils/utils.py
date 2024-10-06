# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import random
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union

import numpy as np
import torch.distributed as dist
from transformers import HfArgumentParser, enable_full_determinism, set_seed
from transformers.trainer import TrainingArguments

from .logger import get_logger
from .np_utils import stat_array
from .torch_utils import broadcast_string, is_dist, is_dist_ta, is_local_master

logger = get_logger()


def get_time_info(log_history: List[Dict[str, Any]], n_train_samples: Optional[int]) -> Optional[Dict[str, Any]]:
    time_info = None
    try:
        last_log_history = log_history[-1]
        train_runtime = last_log_history['train_runtime']
        train_samples_per_second = n_train_samples / train_runtime
        time_info = {
            'train_runtime': train_runtime,
            'n_train_samples': n_train_samples,
            'train_samples_per_second': train_samples_per_second,
        }
    except Exception:
        pass
    return time_info


def check_json_format(obj: Any) -> Any:
    if obj is None or isinstance(obj, (int, float, str, complex)):  # bool is a subclass of int
        return obj

    if isinstance(obj, Sequence):
        res = []
        for x in obj:
            res.append(check_json_format(x))
    elif isinstance(obj, Mapping):
        res = {}
        for k, v in obj.items():
            if 'hub_token' in k:
                res[k] = None
            else:
                if isinstance(v, TrainingArguments):
                    for _k in v.__dict__.keys():
                        if 'hub_token' in _k:
                            setattr(v, _k, None)
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


def _safe_split(s: str,
                sep: str,
                use_0: bool,
                split_mode: Literal['left', 'right'] = 'left') -> Tuple[Optional[str], Optional[str]]:
    # use_0: When the length of the part is 1, is it considered as part0 or part1.
    if s is None or len(s) == 0:
        return None, None
    if split_mode == 'left':
        part = s.split(sep, 1)
    else:
        part = s.rsplit(sep, 1)
    if len(part) == 1:
        if use_0:
            part = part[0], None
        else:
            part = None, part[0]
    else:
        assert len(part) == 2
    return part


def deep_getattr(obj, attr: str, default=None):
    attrs = attr.split('.')
    for a in attrs:
        if obj is None:
            break
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
        sub_folder = broadcast_string(sub_folder)

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


def is_pai_training_job() -> bool:
    return 'PAI_TRAINING_JOB_ID' in os.environ


def get_pai_tensorboard_dir() -> Optional[str]:
    return os.environ.get('PAI_OUTPUT_TENSORBOARD')


def subprocess_run(command: List[str], env: Optional[Dict[str, str]] = None, stdout=None, stderr=None):
    # stdoutm stderr: e.g. subprocess.PIPE.
    resp = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    resp.check_returncode()
    return resp

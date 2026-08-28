# Copyright (c) ModelScope Contributors. All rights reserved.
"""General-purpose helpers, copied from ``swift.utils.utils`` so the dev stack is self-contained."""
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np
import torch
from transformers.utils import strtobool

from .logger import get_logger

logger = get_logger()

_T = TypeVar('_T')


def check_json_format(obj: Any, token_safe: bool = True) -> Any:  # noqa: C901  # verbatim copy of legacy
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
            if token_safe and isinstance(k, str) and '_token' in k and isinstance(v, str):
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


def get_env_args(args_name: str, type_func: Callable[[str], _T], default_value: Optional[_T]) -> Optional[_T]:
    args_name_upper = args_name.upper()
    value = os.getenv(args_name_upper)
    if value is None:
        value = default_value
        log_info = (f'Setting {args_name}: {default_value}. '
                    f'You can adjust this hyperparameter through the environment variable: `{args_name_upper}`.')
    else:
        if type_func is bool:
            value = strtobool(value)
        value = type_func(value)
        log_info = f'Using environment variable `{args_name_upper}`, Setting {args_name}: {value}.'
    logger.info_once(log_info)
    return value


def split_list(ori_list: List[_T], num_shards: int, contiguous=True) -> List[List[_T]]:
    shard = []
    if contiguous:
        idx_list = np.linspace(0, len(ori_list), num_shards + 1, dtype=np.int64)
        for i in range(len(idx_list) - 1):
            shard.append(ori_list[idx_list[i]:idx_list[i + 1]])
    else:
        ori_list = np.array(ori_list)
        for i in range(num_shards):
            shard.append(ori_list[np.arange(i, len(ori_list), num_shards)].tolist())
    return shard


def json_parse_to_dict(value: Union[str, Dict, None], strict: bool = True) -> Union[str, Dict]:
    """Convert a JSON string or JSON file into a dict"""
    # If the value could potentially be a string, it is generally advisable to set strict to False.
    if value is None:
        value = {}
    elif isinstance(value, str):
        if os.path.exists(value):  # local path
            with open(value, 'r', encoding='utf-8') as f:
                value = json.load(f)
        else:  # json str
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                if strict:
                    import json_repair
                    try:
                        # fix malformed json string, e.g., incorrect quotation marks
                        old_value = value
                        value = json_repair.repair_json(value)
                        logger.warning(f'Unable to parse json string, try to repair it, '
                                       f"the string before and after repair are '{old_value}' | '{value}'")
                        value = json.loads(value)
                    except Exception:
                        logger.error(f"Unable to parse json string: '{value}', and try to repair failed")
                        raise
    return value

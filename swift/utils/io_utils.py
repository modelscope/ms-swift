# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, List

import json

from .logger import get_logger
from .utils import check_json_format

logger = get_logger()


def read_from_jsonl(fpath: str, encoding: str = 'utf-8') -> List[Any]:
    res: List[Any] = []
    with open(fpath, 'r', encoding=encoding) as f:
        for line in f:
            res.append(json.loads(line))
    return res


def write_to_jsonl(fpath: str, obj_list: List[Any], encoding: str = 'utf-8') -> None:
    res: List[str] = []
    for obj in obj_list:
        res.append(json.dumps(obj, ensure_ascii=False))
    with open(fpath, 'w', encoding=encoding) as f:
        text = '\n'.join(res)
        f.write(f'{text}\n')


def append_to_jsonl(fpath: str, obj: Any, encoding: str = 'utf-8') -> None:
    obj = check_json_format(obj)
    try:
        with open(fpath, 'a', encoding=encoding) as f:
            f.write(f'{json.dumps(obj, ensure_ascii=False)}\n')
    except Exception as e:
        logger.error(f'Cannot write content to jsonl file:{obj}')
        logger.error(e)

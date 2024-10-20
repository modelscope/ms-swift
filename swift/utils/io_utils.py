# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Union

import json
import requests
from tqdm import tqdm

from .logger import get_logger
from .utils import check_json_format

logger = get_logger()


def download_files(url: str, local_path: str, cookies) -> None:
    resp = requests.get(url, cookies=cookies, stream=True)
    with open(local_path, 'wb') as f:
        for data in tqdm(resp.iter_lines()):
            f.write(data)


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


def append_to_jsonl(fpath: str, obj: Union[Dict, List[Dict]], *, encoding: str = 'utf-8', strict: bool = True) -> None:
    if not isinstance(obj, (list, tuple)):
        obj_list = [obj]
    else:
        obj_list = obj
    obj_list = check_json_format(obj_list)
    try:
        text = ''
        for _obj in obj_list:
            text += f'{json.dumps(_obj, ensure_ascii=False)}\n'
        with open(fpath, 'a', encoding=encoding) as f:
            f.write(text)
    except Exception as e:
        if strict:
            raise
        logger.error(f'Cannot write content to jsonl file. obj: {obj}')
        logger.error(e)

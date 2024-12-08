# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager
from typing import Any, Dict, List, Union

import json
import requests
from modelscope.hub.api import ModelScopeConfig
from tqdm import tqdm

from .logger import get_logger
from .utils import check_json_format

logger = get_logger()


def download_ms_file(url: str, local_path: str, cookies=None) -> None:
    if cookies is None:
        cookies = ModelScopeConfig.get_cookies()
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


class JsonlWriter:

    def __init__(self, fpath: str, *, buffer_size: int = 0, encoding: str = 'utf-8', strict: bool = True):
        self.fpath = fpath
        self.buffer_size = buffer_size
        self.encoding = encoding
        self.strict = strict

        self._cache_text = ''

    def append(self, obj: Union[Dict, List[Dict]]):
        if isinstance(obj, (list, tuple)) and all(isinstance(item, dict) for item in obj):
            obj_list = obj
        else:
            obj_list = [obj]
        obj_list = check_json_format(obj_list)
        for _obj in obj_list:
            self._cache_text += f'{json.dumps(_obj, ensure_ascii=False)}\n'

        if len(self._cache_text) >= self.buffer_size:
            self._write_buffer()

    def close(self):
        self._write_buffer()

    def _write_buffer(self):
        if not self._cache_text:
            return
        try:
            with open(self.fpath, 'a', encoding=self.encoding) as f:
                f.write(self._cache_text)
        except Exception:
            if self.strict:
                raise
            logger.error(f'Cannot write content to jsonl file. cache_text: {self._cache_text}')
        finally:
            self._cache_text = ''


@contextmanager
def open_jsonl_writer(fpath: str, *, buffer_size: int = 0, encoding: str = 'utf-8', strict: bool = True):
    json_writer = JsonlWriter(fpath, buffer_size=buffer_size, encoding=encoding, strict=strict)
    try:
        yield json_writer
    finally:
        json_writer.close()


def append_to_jsonl(fpath: str, obj: Union[Dict, List[Dict]], *, encoding: str = 'utf-8', strict: bool = True) -> None:
    with open_jsonl_writer(fpath, encoding=encoding, strict=strict) as jsonl_writer:
        jsonl_writer.append(obj)

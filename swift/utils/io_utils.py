# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Literal, Union

import json
import requests
import torch.distributed as dist
from accelerate.utils import gather_object
from modelscope.hub.api import ModelScopeConfig
from tqdm import tqdm

from .env import is_master
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

    def __init__(self, fpath: str, *, encoding: str = 'utf-8', strict: bool = True, enable_async: bool = False):
        self.fpath = os.path.abspath(os.path.expanduser(fpath)) if is_master() else None
        self.encoding = encoding
        self.strict = strict
        self.enable_async = enable_async
        self._queue = Queue()
        self._thread = None

    def _append_worker(self):
        while True:
            item = self._queue.get()
            self._append(**item)

    def _append(self, obj: Union[Dict, List[Dict]], gather_obj: bool = False):
        if isinstance(obj, (list, tuple)) and all(isinstance(item, dict) for item in obj):
            obj_list = obj
        else:
            obj_list = [obj]
        if gather_obj and dist.is_initialized():
            obj_list = gather_object(obj_list)
        if not is_master():
            return
        obj_list = check_json_format(obj_list)
        for i, _obj in enumerate(obj_list):
            obj_list[i] = json.dumps(_obj, ensure_ascii=False) + '\n'
        self._write_buffer(''.join(obj_list))

    def append(self, obj: Union[Dict, List[Dict]], gather_obj: bool = False):
        if self.enable_async:
            if self._thread is None:
                self._thread = Thread(target=self._append_worker, daemon=True)
                self._thread.start()
            self._queue.put({'obj': obj, 'gather_obj': gather_obj})
        else:
            self._append(obj, gather_obj=gather_obj)

    def _write_buffer(self, text: str):
        if not text:
            return
        assert is_master(), f'is_master(): {is_master()}'
        try:
            os.makedirs(os.path.dirname(self.fpath), exist_ok=True)
            with open(self.fpath, 'a', encoding=self.encoding) as f:
                f.write(text)
        except Exception:
            if self.strict:
                raise
            logger.error(f'Cannot write content to jsonl file. text: {text}')


def append_to_jsonl(fpath: str, obj: Union[Dict, List[Dict]], *, encoding: str = 'utf-8', strict: bool = True) -> None:
    jsonl_writer = JsonlWriter(fpath, encoding=encoding, strict=strict)
    jsonl_writer.append(obj)


def get_file_mm_type(file_name: str) -> Literal['image', 'video', 'audio']:
    video_extensions = {'.mp4', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.webm'}
    audio_extensions = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    _, ext = os.path.splitext(file_name)

    if ext.lower() in video_extensions:
        return 'video'
    elif ext.lower() in audio_extensions:
        return 'audio'
    elif ext.lower() in image_extensions:
        return 'image'
    else:
        raise ValueError(f'file_name: {file_name}, ext: {ext}')

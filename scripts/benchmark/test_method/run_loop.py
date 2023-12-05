# CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/test_method/run_loop.py

import os
import subprocess
from typing import List

from swift.utils import read_from_jsonl, write_to_jsonl


def test_method_loop(train_kwargs_jsonl: str) -> None:
    while True:
        obj_list = read_from_jsonl(train_kwargs_jsonl)
        if len(obj_list[0]) == 0:
            break
        obj: List[str] = obj_list.pop(0)
        obj_list.append(obj)
        write_to_jsonl(train_kwargs_jsonl, obj_list)
        subprocess.run(
            ['python', 'scripts/benchmark/test_method/run_single.py', *obj])


if __name__ == '__main__':
    jsonl_path = os.path.join('scripts/benchmark/test_method/run.jsonl')
    test_method_loop(jsonl_path)

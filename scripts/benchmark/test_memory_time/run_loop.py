# CUDA_VISIBLE_DEVICES=0 nohup python scripts/benchmark/test_memory_time/run_loop.py &> 0.out &

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import subprocess
from typing import List

from swift.utils import read_from_jsonl, write_to_jsonl


def test_memory_time_loop(train_kwargs_jsonl: str) -> None:
    while True:
        obj_list = read_from_jsonl(train_kwargs_jsonl)
        if len(obj_list[0]) == 0:
            break
        obj: List[str] = obj_list.pop(0)
        obj_list.append(obj)
        write_to_jsonl(train_kwargs_jsonl, obj_list)
        ret = subprocess.run(['python', 'scripts/benchmark/test_memory_time/run_single.py', *obj])
        assert ret.returncode == 0


if __name__ == '__main__':
    jsonl_path = os.path.join('scripts/benchmark/test_memory_time/run.jsonl')
    test_memory_time_loop(jsonl_path)

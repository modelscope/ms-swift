# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, List

import json


def read_from_jsonl(fpath: str, encoding: str = 'utf-8') -> List[Any]:
    res: List[Any] = []
    with open(fpath, 'r', encoding=encoding) as f:
        for line in f:
            res.append(json.loads(line))
    return res

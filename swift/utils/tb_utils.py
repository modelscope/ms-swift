# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, List, Tuple

from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator

Item = Dict[str, float]
TB_COLOR, TB_COLOR_SMOOTH = '#FFE2D9', '#FF7043'


def read_tensorboard_file(fpath: str) -> Dict[str, List[Item]]:
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f'fpath: {fpath}')
    ea = EventAccumulator(fpath)
    ea.Reload()
    res: Dict[str, List[Item]] = {}
    tags = ea.Tags()['scalars']
    for tag in tags:
        values = ea.Scalars(tag)
        r: List[Item] = []
        for v in values:
            r.append({'step': v.step, 'value': v.value})
        res[tag] = r
    return res


def tensorboard_smoothing(values: List[float],
                          smooth: float = 0.9) -> List[float]:
    norm_factor = 1
    x = 0
    res: List[float] = []
    for i in range(len(values)):
        x = x * smooth + values[i]  # Exponential decay
        res.append(x / norm_factor)

        norm_factor *= smooth
        norm_factor += 1
    return res

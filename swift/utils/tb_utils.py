# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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


def tensorboard_smoothing(values: List[float], smooth: float = 0.9) -> List[float]:
    norm_factor = 1
    x = 0
    res: List[float] = []
    for i in range(len(values)):
        x = x * smooth + values[i]  # Exponential decay
        res.append(x / norm_factor)

        norm_factor *= smooth
        norm_factor += 1
    return res


def plot_images(images_dir: str,
                tb_dir: str,
                smooth_key: List[str],
                smooth_val: float = 0.9,
                figsize: Tuple[int, int] = (8, 5),
                dpi: int = 100) -> None:
    """Using tensorboard's data content to plot images"""
    os.makedirs(images_dir, exist_ok=True)
    fname = [fname for fname in os.listdir(tb_dir) if os.path.isfile(os.path.join(tb_dir, fname))][0]
    tb_path = os.path.join(tb_dir, fname)
    data = read_tensorboard_file(tb_path)

    for k in data.keys():
        _data = data[k]
        steps = [d['step'] for d in _data]
        values = [d['value'] for d in _data]
        if len(values) == 0:
            continue
        _, ax = plt.subplots(1, 1, squeeze=True, figsize=figsize, dpi=dpi)
        ax.set_title(k)
        if len(values) == 1:
            ax.scatter(steps, values, color=TB_COLOR_SMOOTH)
        elif k in smooth_key:
            ax.plot(steps, values, color=TB_COLOR)
            values_s = tensorboard_smoothing(values, smooth_val)
            ax.plot(steps, values_s, color=TB_COLOR_SMOOTH)
        else:
            ax.plot(steps, values, color=TB_COLOR_SMOOTH)
        fpath = os.path.join(images_dir, k.replace('/', '_'))
        plt.savefig(fpath, dpi=dpi, bbox_inches='tight')
        plt.close()

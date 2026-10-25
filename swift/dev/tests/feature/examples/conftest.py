# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared fixtures for the examples L2 smoke suites (S16).

The smoke contract, stated once for every group file: one test replays one ``examples/**/*.sh``
(or its yaml twin) with the argv taken VERBATIM from the script -- semantic / legality flags must
survive, because the example is the contract under test -- under exactly three smoke
substitutions (model stays the script's hub id and is served from the local cache; the dataset
becomes a tiny local ``.jsonl``; the training volume collapses to ``--max_steps 3`` with
``--report_to none`` and a tmp ``--output_dir``).
"""
import json
import os

import pytest

#: The hub ids the train examples pin, passed through verbatim (id -> family matching in
#: resolve_loader_cls needs the id; a local snapshot path would fall to an architectures
#: ambiguity). Served from the local MODELSCOPE_CACHE.
MODEL_INSTRUCT = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_BASE = 'Qwen/Qwen2.5-0.5B'
MODEL_THINK = 'Qwen/Qwen3-0.6B'


@pytest.fixture
def sft_rows(tmp_path):
    """Sixteen chat rows standing in for the examples' hub datasets -- small enough to keep a
    smoke fast, real enough that the loss is meaningful on the first pass."""
    rows = []
    for i in range(8):
        rows.append({
            'messages': [
                {'role': 'user', 'content': '你是谁？'},
                {'role': 'assistant', 'content': f'我是 Swift 训练的助手 {i} 号。'},
            ]
        })
        rows.append({
            'messages': [
                {'role': 'user', 'content': '你会什么？'},
                {'role': 'assistant', 'content': f'我会微调大模型，编号 {i}。'},
            ]
        })
    path = tmp_path / 'sft_rows.jsonl'
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


def assert_smoke(history, out_dir, task_type):
    """The example-smoke post-condition: steps ran, losses finite, final checkpoint written.

    The runtime nests a version dir (``v0-<timestamp>``) under ``--output_dir`` (utils
    ``add_version_to_work_dir``), so the final checkpoint is located by glob, not by joining.
    Returns the checkpoint dir for per-example extra assertions.
    """
    assert history, 'recipe produced no optimizer steps'
    losses = [r['loss'] for r in history]
    assert all(loss == loss and abs(loss) != float('inf') for loss in losses), f'non-finite loss: {losses}'
    import glob as _glob
    finals = sorted(_glob.glob(os.path.join(out_dir, 'v*', 'checkpoint-final')))
    assert finals, f'no final checkpoint under {out_dir}: ' \
                   f'{os.listdir(out_dir) if os.path.isdir(out_dir) else "no dir"}'
    ckpt = finals[-1]
    files = set(os.listdir(ckpt))
    assert any(f.endswith('.safetensors') for f in files), f'no weights in {sorted(files)}'
    with open(os.path.join(ckpt, 'args.json')) as f:
        args = json.load(f)
    assert args.get('task_type') == task_type, f'args.json task_type={args.get("task_type")!r}'
    return ckpt

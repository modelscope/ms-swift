# Copyright (c) ModelScope Contributors. All rights reserved.
"""OPSD dataset plugin: derive the privileged ``teacher_prompt`` column from a reference solution.

On-Policy Self-Distillation needs one extra column beyond the dialogue: ``teacher_prompt``. It carries
the PRIVILEGED view the teacher conditions on -- the student's question PLUS a reference solution /
rubric -- while the student itself only ever sees the question. run_opsd replaces the first user turn
with ``teacher_prompt`` when it builds the teacher's scoring view, and keeps the response tokens shared
between teacher and student.

A dataset that already ships a ``teacher_prompt`` column needs no plugin (see examples/v5/rl/data/
opsd.jsonl). This plugin is the pattern for a REAL dataset that only has a ``solution`` / ``answer``
column: it registers a self-contained dataset whose loader builds ``teacher_prompt`` in code, so the
derivation lives next to the data rather than being hand-written into every row.

Loaded with ``--external_plugins examples/v5/rl/opsd/opsd_plugin.py`` and selected with
``--dataset opsd_synthetic`` (the ``dataset_type`` registered below). Swap the in-code ``_ROWS`` for a
hub dataset's rows -- e.g. map each row of ``open-r1/OpenThoughts-114k-math`` or ``AI-MO/NuminaMath-TIR``
through ``_build_row`` -- to distil on real data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import Dataset as HfDataset

from swift.dev.dataset.loader import DatasetLoader, register_dataset

SYSTEM_PROMPT = 'Solve the problem step by step, and put your final answer within \\boxed{}.'

TRANSITION_PROMPT = ('After understanding the reference solution and the rationale behind each step, '
                     'now articulate your own step-by-step reasoning that derives the final answer.')

#: (problem, reference solution) pairs. ``_ROWS`` stands in for a real dataset's problem/solution columns.
_ROWS: List[Dict[str, str]] = [
    {
        'problem': 'What is 12 * 8?',
        'solution': '12 * 8 = 96, so the answer is 96.'
    },
    {
        'problem': 'What is 15 + 27?',
        'solution': '15 + 27 = 42, so the answer is 42.'
    },
    {
        'problem': 'A shop sells pens 3 for $2. How many dollars do 9 pens cost?',
        'solution': '9 pens is 9 / 3 = 3 groups of 3, and each group costs $2, so 3 * 2 = $6.'
    },
    {
        'problem': 'What is the area of a rectangle with sides 5 and 4?',
        'solution': 'Area = length * width = 5 * 4 = 20.'
    },
]


def _build_row(problem: str, solution: Optional[str]) -> Dict[str, Any]:
    """One standard row: the student's ``messages`` (question only) plus the teacher's privileged view.

    ``teacher_prompt`` is the FULL replacement for the first user turn, so it repeats the problem and
    appends the reference solution and a transition instruction. When there is no reference solution the
    row carries no ``teacher_prompt``; run_opsd rejects such a dataset loudly (a teacher that sees exactly
    what the student sees makes the distillation signal identically zero -- that is plain GKD, not OPSD).
    """
    messages = [
        {
            'role': 'system',
            'content': SYSTEM_PROMPT
        },
        {
            'role': 'user',
            'content': problem
        },
    ]
    row: Dict[str, Any] = {'messages': messages}
    if solution:
        row['teacher_prompt'] = (f'{problem}\n\n'
                                 f'Here is a reference solution to this problem:\n{solution}\n\n'
                                 f'{TRANSITION_PROMPT}')
    return row


@register_dataset
class OPSDSyntheticLoader(DatasetLoader):
    """``--dataset opsd_synthetic`` loads these rows with no hub access.

    Rows arrive in the standard ``messages`` layout plus a ``teacher_prompt`` column; the auto-detecting
    format converter passes both through (it keeps unknown columns), and run_opsd reads ``teacher_prompt``
    from each row's extras. Overriding ``build_dataset`` is the hook a self-contained dataset uses.
    """

    dataset_type = 'opsd_synthetic'

    def build_dataset(self, subset, split, **kwargs):
        rows = [_build_row(raw['problem'], raw.get('solution')) for raw in _ROWS]
        return HfDataset.from_list(rows)

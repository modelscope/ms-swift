"""OPSD dataset plugin for open-r1/OpenThoughts-114k-math.

Prepares the dataset for On-Policy Self-Distillation:
- Student sees only the problem.
- Teacher sees the problem + reference solution (privileged info via teacher_prompt).
- Only verified-correct examples are used.

Usage:
    swift rlhf --rlhf_type gkd --external_plugins opsd_plugin.py ...
"""
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset

SYSTEM_PROMPT = 'Please reason step by step, and put your final answer within \\boxed{}.'

TRANSITION_PROMPT = ('After understanding the reference solution and the rationale behind each step, '
                     'now articulate your own step-by-step reasoning that derives the final answer.')


class OpenThoughtsOPSDPreprocessor(RowPreprocessor):
    """Preprocessor that builds teacher_prompt from the reference solution.

    Both student and teacher share the same system prompt for format guidance.
    The teacher's user message additionally includes the reference solution as privileged info.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not row.get('correct', True):
            return None

        problem = row.get('problem', '')
        solution = row.get('solution', '')

        teacher_prompt = (f'{problem}\n\n'
                          f'Here is a reference solution to this problem:\n{solution}\n\n'
                          f'{TRANSITION_PROMPT}')

        messages: List[Dict[str, str]] = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': problem
            },
        ]

        return {'messages': messages, 'teacher_prompt': teacher_prompt}


register_dataset(
    DatasetMeta(
        ms_dataset_id='open-r1/OpenThoughts-114k-math',
        hf_dataset_id='open-r1/OpenThoughts-114k-math',
        preprocess_func=OpenThoughtsOPSDPreprocessor(),
        tags=['math', 'opsd'],
    ))

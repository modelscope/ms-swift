# Copyright (c) ModelScope Contributors. All rights reserved.
# One batch's prompts and completions, read off the trajectories for the tracker and completions.jsonl.
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function
from twinkle.data_format import last_assistant_text


@remote_class(execute='first')
class CompletionLog:
    """``(prompt messages, completion text)`` per trajectory; with ``path``, also append them to a JSONL.

    The trajectories carry the messages (references under ray), so the rows are read where they
    live and only the text comes back. Each JSONL line holds the step, the prompt, the completion
    and its reasoning, the full messages, the completion length and the trajectory's ``truncated``
    flag and named ``rewards.orm`` column.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def __call__(self, trajs: Sequence[Dict[str, Any]], *, step: int,
                 path: Optional[str] = None, **kwargs) -> List[Tuple[List[dict], str]]:
        rows = []
        lines = []
        for traj in trajs:
            messages = list(traj.get('messages') or [])
            first = next((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), len(messages))
            completion = last_assistant_text(traj)
            rows.append((messages[:first], completion))
            if path is not None:
                assistant = next((m for m in reversed(messages) if m.get('role') == 'assistant'), {})
                lines.append(json.dumps({
                    'step': step,
                    'prompt': messages[:first],
                    'completion': completion,
                    'reasoning_content': assistant.get('reasoning_content'),
                    'messages': messages,
                    'completion_length': len(traj.get('response_token_ids') or []),
                    'truncated': bool(traj.get('truncated')),
                    'rewards': ((traj.get('rewards') or {}).get('orm') or {}),
                }, ensure_ascii=False, default=str))
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as stream:
                stream.writelines(line + '\n' for line in lines)
        return rows

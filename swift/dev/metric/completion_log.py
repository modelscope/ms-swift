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
    flag and reward accounting -- the named ``rewards.orm`` column (the score rows ``scores``,
    row-aligned, as the driver hands them out beside the trajectories, when given; the trajectory's
    own column otherwise) plus the rollout-owned ``rewards.env`` when the trajectory carries it.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def __call__(self, trajs: Sequence[Dict[str, Any]], *, step: int,
                 path: Optional[str] = None, scores: Optional[Sequence[Dict[str, Any]]] = None,
                 **kwargs) -> List[Tuple[List[dict], str]]:
        if scores is not None and len(scores) != len(trajs):
            raise ValueError(f'scores carries {len(scores)} rows for {len(trajs)} trajectories.')
        rows = []
        lines = []
        for index, traj in enumerate(trajs):
            messages = list(traj.get('messages') or [])
            first = next((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), len(messages))
            completion = last_assistant_text(traj)
            rows.append((messages[:first], completion))
            if path is not None:
                assistant = next((m for m in reversed(messages) if m.get('role') == 'assistant'), {})
                source = scores[index] if scores is not None else traj
                # The rollout owns rewards.env (the sandbox's own reward, stamped at episode
                # finish); the scorer owns the orm names. The audit line must account BOTH --
                # writing only the orm column silently dropped the env channel, so env/training
                # reward accounting could not be verified from the run artifacts. The flat orm
                # schema is unchanged; 'env' joins it only when the trajectory carries it.
                traj_rewards = traj.get('rewards') or {}
                rewards = dict((source.get('rewards') or {}).get('orm') or {})
                if 'env' in traj_rewards:
                    rewards['env'] = traj_rewards['env']
                lines.append(json.dumps({
                    'step': step,
                    'prompt': messages[:first],
                    'completion': completion,
                    'reasoning_content': assistant.get('reasoning_content'),
                    'messages': messages,
                    'completion_length': len(traj.get('response_token_ids') or []),
                    'truncated': bool(traj.get('truncated')),
                    'rewards': rewards,
                }, ensure_ascii=False, default=str))
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as stream:
                stream.writelines(line + '\n' for line in lines)
        return rows

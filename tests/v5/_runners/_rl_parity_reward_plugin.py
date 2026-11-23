import json
import os

from swift.rewards.orm import orms


class GrpoContentReward:
    def __init__(self, args=None, **kwargs):
        pass

    def __call__(self, completions, **kwargs):
        def _score(text):
            return 1.0 if 'step by step' in text.lower() else 0.1
        dump = os.environ.get('RL_PARITY_DUMP')
        if dump:
            with open(dump, 'a', encoding='utf-8') as f:
                f.write(json.dumps(
                    {'prompt_side': [{'k': k, 'v': str(v)[:200]} for k, v in kwargs.items()]},
                    ensure_ascii=False) + chr(10))
                for text in completions:
                    f.write(json.dumps({'len': len(text), 'text': text},
                                       ensure_ascii=False) + chr(10))
        return [_score(text) for text in completions]


orms['grpo_content_reward'] = GrpoContentReward

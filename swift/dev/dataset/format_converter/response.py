# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
from typing import Any, Collection, Dict, Optional

from .base import FormatConverter, register_format

__all__ = ['ResponseConverter']


@register_format
class ResponseConverter(FormatConverter):
    """Flat ``query``/``response`` rows, with optional multi-turn ``history`` and preference pair."""

    format_name = 'response'
    priority = 1000
    is_fallback = True
    aliases = {
        'system_prompt': 'system',
        'prompt': 'query',
        'input': 'query',
        'instruction': 'query',
        'question': 'query',
        'problem': 'query',
        'answer': 'response',
        'answer_key': 'response',
        'answers': 'response',
        'output': 'response',
        'targets': 'response',
        'target': 'response',
        'solution': 'response',
        'text': 'response',
        'completion': 'response',
        'content': 'response',
    }

    @classmethod
    def detect(cls, columns: Collection[str]) -> bool:
        standardised = {cls.aliases.get(column, column) for column in columns}
        ## 这里的detect发生在aliases前还是后？ 会不会有alias所以检测不到
        # 另外一个response够吗
        return 'response' in standardised

    def convert(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.apply_aliases(row)
        response = self.pick_response(row.pop('response', None))
        query = row.pop('query', None)
        system = row.pop('system', None)

        history = self.parse_literal(row.pop('history', None)) or []
        # The current turn is the last turn; a None response is a valid unanswered final turn.
        turns = [*history, [query, response]]
        messages = self.history_to_messages(turns, system)
        if not messages:
            return None
        row['messages'] = messages
        rejected = self.pick_response(row.pop('rejected_response', None))
        if rejected is not None:
            row['rejected_response'] = rejected
        return row

    def pick_response(self, response: Any) -> Any:
        """Collapse a multi-answer response column to a single answer.

        Some datasets store several reference answers in a list. Default to the first for
        reproducibility; ``RANDOM_DATASET_RESPONSE=1`` picks one at random, matching legacy's env
        switch (used when the extra answers are meant as augmentation, not alternatives).
        """
        if not isinstance(response, (list, tuple)):
            return response
        if not response:
            return None
        ## 这个直接源代码拷贝过来吧，没必要依赖transformers
        from transformers.utils import strtobool
        ## 新建一个env的py，管理所有的环境变量
        ## 使用的时候env_utils.pick_random_response()就行。扫描一下，所有的类似的地方都这么处理
        if strtobool(os.environ.get('RANDOM_DATASET_RESPONSE', 'False')):
            ## 放置在文件头。所有的重依赖放在局部，轻依赖放在文件头
            import random
            return random.choice(response)
        return response[0]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from swift.llm import Messages


def messages_join_observation(messages: Messages):
    """
        Joins observations from 'tool' message into the 'assistant' response.

        Example:
        ---------
        Original messages:
        messages = [
            {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
            {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                  [{"location": "Hangzhou"}]\nObservations:'},
            {'role': 'tool', 'content': 'It is 26 degrees Celsius and sunny in Hangzhou today.'}
        ]

        Transformed messages:
        messages = [
            {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
            {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                  [{"location": "Hangzhou"}]\nObservations: It is 26 degrees Celsius and sunny in Hangzhou today.'}
        ]
        """

    if len(messages) >= 2 and messages[-2]['role'] == 'assistant' and messages[-2]['content'] and messages[-2][
            'content'].endswith('Observation:'):
        assert messages[-1]['role'] == 'tool'
        observations = messages[-1]['content']
        messages.pop(-1)
        messages[-1]['content'] += observations
    return

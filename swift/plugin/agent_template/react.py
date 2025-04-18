# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Optional, Union

import json

from .base import BaseAgentTemplate


class ReactEnAgentTemplate(BaseAgentTemplate):

    def format_system(self, tool_names: List[str], tools: List[Union[str, Dict[str, Any]]],
                      system: Optional[str]) -> str:
        tools = [t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in tools]
        return """Answer the following questions as best you can. You have access to the following tools:

""" + '\n\n'.join(tools) + f"""

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{','.join(tool_names)}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

    def format_observations(self, observations: List[str]) -> str:
        res = []
        for observation in observations:
            res += [observation, '\n']
        return ''.join(res)

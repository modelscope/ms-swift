# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Tuple

from swift.utils import get_logger
from swift.utils.utils import split_str_parts_by

logger = get_logger()


def calculate_loss_scale(response: str,
                         use_loss_scale=False,
                         loss_scale_map: dict[str:list] = None) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.

    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf

    Agent response format:

    ```text
        Thought: you should always think about what to do
        Action: the action to take, should be one of the above tools[fire_recognition,
            fire_alert, call_police, call_fireman]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ```

    Args:
        response: The response text
        use_loss_scale: Use weighted loss. With this, some part of the loss will be enhanced to improve performance.

    Returns:
        A tuple of agent response parts and their weights.
    """
    if use_loss_scale:
        agent_parts = split_str_parts_by(response, loss_scale_map)
        weights = []
        agent_content = []
        for c in agent_parts:
            if isinstance(c['key'], (float, int)):
                weights += [c['key']]
                agent_content.append(c['content'])
            else:
                if c['key'] in loss_scale_map:
                    weights += [loss_scale_map[c['key']][0]]
                    weights += [loss_scale_map[c['key']][1]]
                else:
                    weights += [1.0]
                    weights += [1.0]
                agent_content.append(c['key'])
                agent_content.append(c['content'])
        return agent_content, weights
    else:
        return [response], [1.0]

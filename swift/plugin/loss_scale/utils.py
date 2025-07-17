from typing import Dict, List, Optional, Tuple

from swift.llm.template import split_str_parts_by


def calculate_loss_scale(query: str,
                         response: str,
                         response_loss_scale_map: Dict[str, list],
                         query_loss_scale_map: Optional[Dict[str, list]] = None) -> Tuple[List[str], List[float]]:
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
    Returns:
        A tuple of agent response parts and their weights.
    """
    # query loss scale map
    if query_loss_scale_map is not None:
        for key in query_loss_scale_map.keys():
            if key in query:
                if isinstance(query_loss_scale_map[key], (float, int)):
                    query_loss_scale_map[key] = [query_loss_scale_map[key]]
                loss_scale_value = query_loss_scale_map[key][0]
                return [response], [float(loss_scale_value)]
    delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 2]
    if delimiters:
        agent_parts = split_str_parts_by(response, delimiters)
    else:
        regex_delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 1]
        agent_parts = split_str_parts_by(response, regex_delimiters, regex_mode=True)
    weights = []
    agent_content = []
    for c in agent_parts:
        if c['key'] in response_loss_scale_map:
            loss_scale = response_loss_scale_map[c['key']]
            assert len(loss_scale) in {1, 2}, f'loss_scale: {loss_scale}'
            if len(loss_scale) == 1:
                weights += loss_scale
                agent_content.append(c['content'])
            else:
                weights += loss_scale
                agent_content += [c['key'], c['content']]
        else:
            weights.append(1.)
            agent_content.append(c['content'])
    return agent_content, weights

# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, List, Optional, Tuple

from ..utils import split_str_parts_by


def calculate_loss_scale(
    query: str,
    response: str,
    response_loss_scale_map: Dict[str, list],
    query_loss_scale_map: Optional[Dict[str, list]] = None,
) -> Tuple[List[str], List[float]]:
    """Weigh agent-response segments by delimiter (paper: https://arxiv.org/pdf/2309.00986.pdf).

    Response format: ``Thought: / Action: / Action Input: / Observation:`` blocks ending in
    ``Final Answer:``. Map entries: ``[w]`` = regex-matched chunk weighs w; ``[w_tag, w_content]``
    = literal tag then its content; a query-map hit replaces the whole response weight. Unmatched
    text weighs 1.0. Returns ``(parts, weights)``.
    """
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
        if c["key"] in response_loss_scale_map:
            loss_scale = response_loss_scale_map[c["key"]]
            assert len(loss_scale) in {1, 2}, f"loss_scale: {loss_scale}"
            if len(loss_scale) == 1:
                weights += loss_scale
                agent_content.append(c["content"])
            else:
                weights += loss_scale
                agent_content += [c["key"], c["content"]]
        else:
            weights.append(1.0)
            agent_content.append(c["content"])
    return agent_content, weights

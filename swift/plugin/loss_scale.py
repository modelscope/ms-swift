# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict, List, Optional, Tuple

import json

from swift.llm import Messages
from swift.llm.template.utils import ContextType, split_parts_by_regex, split_str_parts_by


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
    delimiters = list(k for k in response_loss_scale_map.keys() if len(response_loss_scale_map[k]) == 2)
    agent_parts = split_str_parts_by(response, delimiters)
    regex_delimiters = {k: v for k, v in response_loss_scale_map.items() if len(v) == 1}
    if len(regex_delimiters):
        split_parts_by_regex(agent_parts, regex_delimiters)
    weights = []
    agent_content = []
    for c in agent_parts:
        if isinstance(c['key'], (float, int)):
            weights += [c['key']]
            agent_content.append(c['content'])
        else:
            if c['key'] in response_loss_scale_map:
                weights += [response_loss_scale_map[c['key']][0]]
                weights += [response_loss_scale_map[c['key']][1]]
                agent_content.append(c['key'])
                agent_content.append(c['content'])
            else:
                weights += [1.0]
                agent_content.append(c['content'])
    return agent_content, weights


class LossScale:
    loss_scale_config = None  # path

    def __init__(self):
        if self.loss_scale_config is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, 'agent', self.loss_scale_config)
            with open(config_path, 'r', encoding='utf-8') as json_file:
                self.loss_scale_map = json.load(json_file)
        else:
            self.loss_scale_map = None

    def get_loss_scale(self,
                       context: str,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None) -> Tuple[List[str], List[float]]:
        """Calculate loss scale

        Args:
            context: The input context
            context_type: The type of this context, like response/suffix(eos token)/other(query/system, etc.)
            is_last_round: If this is the last round of messages.
            query: The query of this round.

        Returns:
            A tuple, list of context and list of loss_scales
        """
        if context_type in {ContextType.RESPONSE, ContextType.SUFFIX}:
            loss_scale = 1.
        else:
            loss_scale = 0.
        return [context], [loss_scale]

    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,
                 **kwargs) -> Tuple[List[str], List[float]]:
        res_context_list = []
        res_loss_scale = []
        i = 0
        n_round = len(messages) // 2
        for context, context_type in zip(context_list, context_types):
            is_last_round = i + 1 == n_round
            if context_type == ContextType.RESPONSE:
                query = messages[2 * i]['content']
                assert context == messages[2 * i + 1]['content']
                kwargs = {'query': query}
                i += 1
            new_context, loss_scale = self.get_loss_scale(context, context_type, is_last_round, **kwargs)
            res_context_list += new_context
            res_loss_scale += loss_scale
        return res_context_list, res_loss_scale


class LastRoundLossScale(LossScale):

    def get_loss_scale(self, context: str, context_type: ContextType, is_last_round: bool, **kwargs):
        if context_type == ContextType.RESPONSE:
            return [context], [float(is_last_round)]
        return super().get_loss_scale(context, context_type, is_last_round)


class AgentFlanLossScale(LossScale):
    loss_scale_config = 'agentflan.json'

    def get_loss_scale(self,
                       context: str,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None):
        if context_type == ContextType.RESPONSE:
            query_loss_scale_map = self.loss_scale_map['query']
            response_loss_scale_map = self.loss_scale_map['response']
            return calculate_loss_scale(query, context, response_loss_scale_map, query_loss_scale_map)
        return super().get_loss_scale(context, context_type, is_last_round)


class REACTLossScale(LossScale):
    loss_scale_config = 'default_loss_scale_config.json'

    def get_loss_scale(self,
                       context: str,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None):
        if context_type == ContextType.RESPONSE:
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context, context_type, is_last_round)


class QwenLossScale(LossScale):
    loss_scale_config = 'qwen_loss_scale_config.json'

    def get_loss_scale(self,
                       context: str,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None):
        if context_type == ContextType.RESPONSE:
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context, context_type, is_last_round)


class AlphaUmiLossScale(REACTLossScale):
    loss_scale_config = 'alpha_umi_loss_scale_config.json'


class TrainAllLossScale(LossScale):

    def get_loss_scale(self, context: str, context_type: ContextType, *args, **kwargs):
        return [context], [1.]


# Add your loss scale here, use --loss_scale xxx to train
loss_scale_map = {
    'last_round': LastRoundLossScale(),
    'default': LossScale(),
    'all': TrainAllLossScale(),
    # agent
    'agentflan': AgentFlanLossScale(),
    'react': REACTLossScale(),
    'alpha_umi': AlphaUmiLossScale(),
    'qwen': QwenLossScale(),
}

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Tuple, Union

import json

from .utils import calculate_loss_scale


class LossScale:

    SYSTEM = 'system'
    QUERY = 'query'
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    CHAT_SEP = 'chat_sep'
    ROUND = 'round'
    BOS = 'bos'

    def __call__(self, round: int, content: List[Union[str, int]], types: List[str],
                 **kwargs) -> Tuple[List[Union[str, int]], List[float]]:
        if types == [LossScale.RESPONSE]:
            return content, [1.0] * len(content)
        elif types == [LossScale.SUFFIX]:
            return content, [1.0] * len(content)
        else:
            return content, [0.0] * len(content)


class TrainAllLossScale:

    SYSTEM = 'system'
    QUERY = 'query'
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    CHAT_SEP = 'chat_sep'
    BOS = 'bos'

    def __call__(self, round: int, content: List[Union[str, int]], types: List[str], **kwargs):
        return content, [1.0] * len(content)


class AgentFlanLossScale(LossScale):

    def __call__(self, round: int, content: List[Union[str, int]], types: List[str], **kwargs):
        if types == [LossScale.RESPONSE]:
            query = kwargs['query']
            response = kwargs['response']
            assert content == [response]
            loss_scale_config_path = 'agentflan.json'
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, loss_scale_config_path)
            with open(config_path, 'r') as json_file:
                loss_scale_map = json.load(json_file)
            query_loss_scale_map = loss_scale_map['query']
            response_loss_scale_map = loss_scale_map['response']
            return calculate_loss_scale(query, response, response_loss_scale_map, query_loss_scale_map)
        elif types == [LossScale.SUFFIX]:
            return content, [1.0] * len(content)
        else:
            return content, [0.0] * len(content)


class REACTLossScale(LossScale):

    def __call__(self, round: int, content: List[Union[str, int]], types: List[str], **kwargs):
        if types == [LossScale.RESPONSE]:
            query = kwargs['query']
            response = kwargs['response']
            assert content == [response]
            loss_scale_config_path = 'default_loss_scale_config.json'
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, loss_scale_config_path)
            with open(config_path, 'r') as json_file:
                loss_scale_map = json.load(json_file)
            return calculate_loss_scale(query, response, loss_scale_map)
        elif types == [LossScale.SUFFIX]:
            return content, [1.0] * len(content)
        else:
            return content, [0.0] * len(content)


class AlphaUmiLossScale(LossScale):

    def __call__(self, round: int, content: List[Union[str, int]], types: List[str], **kwargs):
        if types == [LossScale.RESPONSE]:
            query = kwargs['query']
            response = kwargs['response']
            assert content == [response]
            loss_scale_config_path = 'alpha_umi_loss_scale_config.json'
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, loss_scale_config_path)
            with open(config_path, 'r') as json_file:
                loss_scale_map = json.load(json_file)
            return calculate_loss_scale(query, response, loss_scale_map)
        elif types == [LossScale.SUFFIX]:
            return content, [1.0] * len(content)
        else:
            return content, [0.0] * len(content)


class LastRoundLossScale(LossScale):

    def __call__(self, round: int, content: List[Union[str, int]], types: List[str], **kwargs):
        if types == [LossScale.RESPONSE] and round + 1 == kwargs.get('n_round', round + 1):
            return content, [1.0] * len(content)
        elif types == [LossScale.SUFFIX]:
            return content, [1.0] * len(content)
        else:
            return content, [0.0] * len(content)


loss_scale_map = {
    'agentflan': AgentFlanLossScale(),
    'react': REACTLossScale(),
    'alpha_umi': AlphaUmiLossScale(),
    'default': LossScale(),
    'all': TrainAllLossScale(),
    'last_round': LastRoundLossScale(),
}

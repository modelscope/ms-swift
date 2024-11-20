# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Optional

import json

from ..template_inputs import Messages
from ..utils import ContextType
from .utils import calculate_loss_scale


class LossScale:
    loss_scale_config = None  # path

    def __init__(self):
        if self.loss_scale_config is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, self.loss_scale_config)
            with open(config_path, 'r') as json_file:
                self.loss_scale_map = json.load(json_file)
        else:
            self.loss_scale_map = None

    def get_loss_scale(self,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None,
                       response: Optional[str] = None):
        if context_type in {ContextType.RESPONSE, ContextType.SUFFIX}:
            return 1.
        else:
            return 0.

    def __call__(self, context_types: List[ContextType], messages: Messages, **kwargs) -> List[float]:
        res = []
        i = 0
        n_round = len(messages) // 2
        for context_type in context_types:
            is_last_round = i + 1 == n_round
            if context_type == ContextType.RESPONSE:
                query, response = messages[2 * i]['content'], messages[2 * i + 1]['content']
                res.append(self.get_loss_scale(context_type, is_last_round, query=query, response=response))
                i += 1
            else:
                res.append(self.get_loss_scale(context_type, is_last_round))
        return res


class TrainAllLossScale(LossScale):

    def get_loss_scale(self, context_type: ContextType, *args, **kwargs):
        return 1.


class LastRoundLossScale(LossScale):

    def get_loss_scale(self, context_type: ContextType, is_last_round: bool, **kwargs):
        if context_type == ContextType.RESPONSE:
            return float(is_last_round)
        return super().get_loss_scale(context_type, is_last_round)


class AgentFlanLossScale(LossScale):
    loss_scale_config = 'agentflan.json'

    def get_loss_scale(self,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None,
                       response: Optional[str] = None):
        if context_type == ContextType.RESPONSE:
            query_loss_scale_map = self.loss_scale_map['query']
            response_loss_scale_map = self.loss_scale_map['response']
            return calculate_loss_scale(query, response, response_loss_scale_map, query_loss_scale_map)
        return super().get_loss_scale(context_type, is_last_round)


class REACTLossScale(LossScale):
    loss_scale_config_path = 'default_loss_scale_config.json'

    def get_loss_scale(self,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None,
                       response: Optional[str] = None):
        if context_type == ContextType.RESPONSE:
            return calculate_loss_scale(query, response, self.loss_scale_map)
        return super().get_loss_scale(context_type, is_last_round)


class AlphaUmiLossScale(REACTLossScale):
    loss_scale_config_path = 'alpha_umi_loss_scale_config.json'


loss_scale_map = {
    'agentflan': AgentFlanLossScale(),
    'react': REACTLossScale(),
    'alpha_umi': AlphaUmiLossScale(),
    'default': LossScale(),
    'all': TrainAllLossScale(),
    'last_round': LastRoundLossScale(),
}

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Optional, Tuple

import json

from swift.llm import Messages
from swift.llm.template.utils import ContextType
from .utils import calculate_loss_scale


class LossScale:
    # Indicates whether loss_scale contains only 0 and 1.
    # If set to True, loss_scale will be replaced by labels to stay compatible with
    # acceleration techniques such as liger_kernel.
    # If set to False, an additional 'loss_scale' key will be stored and the
    # corresponding loss function will be used.
    loss_scale_config = None  # path
    is_binary = None

    def __init__(self):
        if self.loss_scale_config is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, 'config', self.loss_scale_config)
            with open(config_path, 'r', encoding='utf-8') as json_file:
                self.loss_scale_map = json.load(json_file)
        else:
            self.loss_scale_map = None

    def get_loss_scale(self, context: str, context_type: ContextType, is_last_round: bool,
                       **kwargs) -> Tuple[List[str], List[float]]:
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
            query, loss = None, None
            if context_type == ContextType.RESPONSE:
                query = messages[2 * i]['content']
                # Currently, we only support applying loss/mask to the response part.
                loss = messages[2 * i + 1].get('loss')
                assert context == messages[2 * i + 1]['content']
                i += 1
            if isinstance(context, dict) and 'loss_scale' in context:
                new_context = [[token] for token in context['token_ids']]
                loss_scale = context['loss_scale']
            else:
                if isinstance(context, dict) and 'token_ids' in context:
                    context = context['token_ids']
                if context_type == ContextType.RESPONSE and loss is not None:
                    new_context, loss_scale = [context], [float(loss)]
                else:
                    new_context, loss_scale = self.get_loss_scale(context, context_type, is_last_round, query=query)
            res_context_list += new_context
            res_loss_scale += loss_scale
        return res_context_list, res_loss_scale

    @property
    def is_loss_scale_binary(self):
        if self.is_binary is not None:
            return self.is_binary
        if self.loss_scale_map is None:
            return True
        return all(scale == 0.0 or scale == 1.0 for lst in self.loss_scale_map.values() for scale in lst)


class DefaultLossScale(LossScale):
    pass


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
        if context_type == ContextType.RESPONSE and isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map['response'], self.loss_scale_map['query'])
        return super().get_loss_scale(context, context_type, is_last_round)


class REACTLossScale(LossScale):
    loss_scale_config = 'react.json'

    def get_loss_scale(self,
                       context: str,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None):
        if context_type == ContextType.RESPONSE and isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context, context_type, is_last_round)


class QwenLossScale(REACTLossScale):
    loss_scale_config = 'qwen.json'


class HermesLossScale(REACTLossScale):
    loss_scale_config = 'hermes.json'


class AlphaUmiLossScale(REACTLossScale):
    loss_scale_config = 'alpha_umi.json'


class TrainAllLossScale(LossScale):

    def get_loss_scale(self, context: str, context_type: ContextType, *args, **kwargs):
        return [context], [1.]


class IgnoreEmptyThink(REACTLossScale):
    loss_scale_config = 'ignore_empty_think.json'


class LastRoundWithIgnoreEmptyThink(LossScale):
    loss_scale_config = 'ignore_empty_think.json'

    def get_loss_scale(self,
                       context: str,
                       context_type: ContextType,
                       is_last_round: bool,
                       *,
                       query: Optional[str] = None):
        if context_type == ContextType.RESPONSE:
            if not is_last_round:
                return [context], [0.]
            elif isinstance(context, str):
                return calculate_loss_scale(query, context, self.loss_scale_map)

        return super().get_loss_scale(context, context_type, is_last_round)


# Add your loss scale here, use --loss_scale xxx to train
loss_scale_map = {
    'last_round': LastRoundLossScale,
    'default': DefaultLossScale,
    'all': TrainAllLossScale,
    'ignore_empty_think': IgnoreEmptyThink,
    'last_round_with_ignore_empty_think': LastRoundWithIgnoreEmptyThink,
    # agent
    'react': REACTLossScale,
    'hermes': HermesLossScale,
    'qwen': QwenLossScale,
    'agentflan': AgentFlanLossScale,
    'alpha_umi': AlphaUmiLossScale,
}

for k, v in loss_scale_map.items():
    v.name = k

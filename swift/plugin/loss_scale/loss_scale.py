# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Literal, Optional, Tuple

import json

from swift.llm import Messages
from swift.llm.template import get_last_user_round
from swift.llm.template.utils import ContextType
from .utils import calculate_loss_scale

ALL_BASE_STRATEGY = ['default', 'last_round', 'all']


class LossScale:
    # Indicates whether loss_scale contains only 0 and 1.
    # If set to True, loss_scale will be replaced by labels to stay compatible with
    # acceleration techniques such as liger_kernel.
    # If set to False, an additional 'loss_scale' key will be stored and the
    # corresponding loss function will be used.
    loss_scale_config = None  # path
    is_binary = None

    def __init__(self, base_strategy: Literal['default', 'last_round', 'all'] = 'default'):
        assert base_strategy in ALL_BASE_STRATEGY, (
            f'ALL_BASE_STRATEGY: {ALL_BASE_STRATEGY}, base_strategy: {base_strategy}')
        self.base_strategy = base_strategy
        self.loss_scale_map = None
        if self.loss_scale_config is not None:
            path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(path, 'config', self.loss_scale_config)
            with open(config_path, 'r', encoding='utf-8') as json_file:
                self.loss_scale_map = json.load(json_file)

    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        """Calculate loss scale

        Args:
            context: The input context
            query: The query of this round.

        Returns:
            A tuple, list of context and list of loss_scales
        """
        return [context], [1.]

    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,
                 **kwargs) -> Tuple[List[str], List[float]]:
        res_context_list = []
        res_loss_scale = []
        i = 0
        last_user_round = get_last_user_round(messages)
        for context, context_type in zip(context_list, context_types):
            is_last_round = 2 * i >= last_user_round
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
                is_assistant = context_type in {ContextType.RESPONSE, ContextType.SUFFIX}
                if loss or loss is None and (self.base_strategy == 'all' or
                                             (self.base_strategy == 'default' and is_assistant) or
                                             (self.base_strategy == 'last_round' and is_assistant and is_last_round)):
                    new_context, loss_scale = self.get_loss_scale(context, query=query)
                else:
                    new_context, loss_scale = [context], [0.]
            res_context_list += new_context
            res_loss_scale += loss_scale
        # The values in loss_scale_list correspond one-to-one with the values in context_list.
        return res_context_list, res_loss_scale

    @property
    def is_loss_scale_binary(self):
        if self.is_binary is not None:
            return self.is_binary
        if self.loss_scale_map is None:
            return True
        return all(scale in {0.0, 1.0} for lst in self.loss_scale_map.values() for scale in lst)


class AgentFlanLossScale(LossScale):
    loss_scale_config = 'agentflan.json'

    def get_loss_scale(self, context: str, *, query: Optional[str] = None):
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map['response'], self.loss_scale_map['query'])
        return super().get_loss_scale(context)


class REACTLossScale(LossScale):
    loss_scale_config = 'react.json'

    def get_loss_scale(self, context: str, *, query: Optional[str] = None):
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context)


class QwenLossScale(REACTLossScale):
    loss_scale_config = 'qwen.json'


class HermesLossScale(REACTLossScale):
    loss_scale_config = 'hermes.json'


class AlphaUmiLossScale(REACTLossScale):
    loss_scale_config = 'alpha_umi.json'


class IgnoreEmptyThinkLossScale(REACTLossScale):
    loss_scale_config = 'ignore_empty_think.json'


# Add your loss scale here, use --loss_scale xxx to train
loss_scale_map = {
    '-': LossScale,
    'ignore_empty_think': IgnoreEmptyThinkLossScale,
    # agent
    'react': REACTLossScale,
    'hermes': HermesLossScale,
    'qwen': QwenLossScale,
    'agentflan': AgentFlanLossScale,
    'alpha_umi': AlphaUmiLossScale,
}


def get_loss_scale(loss_scale: str) -> LossScale:
    splited = loss_scale.split('+', 1)
    if len(splited) == 1:
        if splited[0] in ALL_BASE_STRATEGY:
            base_strategy, loss_scale = splited[0], '-'
        else:
            base_strategy, loss_scale = 'default', splited[0]
    else:
        base_strategy, loss_scale = splited
    return loss_scale_map[loss_scale](base_strategy)

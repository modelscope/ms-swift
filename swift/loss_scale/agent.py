# Copyright (c) ModelScope Contributors. All rights reserved.

from typing import Optional

from .base import ConfigLossScale
from .utils import calculate_loss_scale


class AgentFlanLossScale(ConfigLossScale):
    is_binary = False
    loss_scale_config = 'agentflan.json'

    def get_loss_scale(self, context: str, *, query: Optional[str] = None):
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map['response'], self.loss_scale_map['query'])
        return super().get_loss_scale(context)


class REACTLossScale(ConfigLossScale):
    loss_scale_config = 'react.json'


class QwenLossScale(ConfigLossScale):
    loss_scale_config = 'qwen.json'


class HermesLossScale(ConfigLossScale):
    loss_scale_config = 'hermes.json'


class AlphaUmiLossScale(ConfigLossScale):
    loss_scale_config = 'alpha_umi.json'

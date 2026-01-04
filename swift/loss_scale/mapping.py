# Copyright (c) Alibaba, Inc. and its affiliates.
from .agent import AgentFlanLossScale, AlphaUmiLossScale, HermesLossScale, QwenLossScale, REACTLossScale
from .base import ALL_BASE_STRATEGY, LossScale
from .other import IgnoreEmptyThinkLossScale

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

# Copyright (c) ModelScope Contributors. All rights reserved.
from .agent import AgentFlanLossScale, AlphaUmiLossScale, HermesLossScale, QwenLossScale, REACTLossScale
from .base import ALL_BASE_STRATEGY, LossScale
from .other import IgnoreEmptyThinkLossScale

# Add your loss scale here, use --loss_scale xxx to train
loss_scale_map = {
    'base': LossScale,
    'ignore_empty_think': IgnoreEmptyThinkLossScale,
    # agent
    'react': REACTLossScale,
    'hermes': HermesLossScale,
    'qwen': QwenLossScale,
    'agentflan': AgentFlanLossScale,
    'alpha_umi': AlphaUmiLossScale,
}


def get_loss_scale(loss_scale: str) -> LossScale:
    """Factory function to create a loss scale object from a string specification.

    The loss_scale string can be in three formats:
    1. A strategy name alone (e.g., 'default', 'last_round', 'all') - uses base LossScale
    2. A loss scale type alone (e.g., 'hermes', 'react') - uses 'default' strategy
    3. A strategy name followed by a loss scale type (e.g., 'default+react', 'last_round+qwen')

    Args:
        loss_scale: String specifying the loss scale configuration. Can be:
            - A base strategy name: 'default', 'last_round', or 'all'
            - A loss scale type: 'base', 'react', 'hermes', 'qwen', etc.
            - A combination: 'base_strategy+loss_scale_type'

    Returns:
        LossScale: An instance of the appropriate LossScale subclass configured
            with the specified base strategy.

    Examples:
        >>> get_loss_scale('default')  # Uses default strategy with base LossScale
        >>> get_loss_scale('react')  # Uses default strategy with REACTLossScale
        >>> get_loss_scale('last_round+hermes')  # Uses last_round strategy with HermesLossScale
    """
    splited = loss_scale.split('+', 1)
    if len(splited) == 1:
        if splited[0] in ALL_BASE_STRATEGY:
            base_strategy, loss_scale = splited[0], 'base'
        else:
            base_strategy, loss_scale = 'default', splited[0]
    else:
        base_strategy, loss_scale = splited
    return loss_scale_map[loss_scale](base_strategy)

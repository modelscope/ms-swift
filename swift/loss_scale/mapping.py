# Copyright (c) ModelScope Contributors. All rights reserved.
from .agent import AgentFlanLossScale, AlphaUmiLossScale, HermesLossScale, QwenLossScale, REACTLossScale
from .base import ALL_BASE_STRATEGY, ConcatLossScale, LossScale
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

    The loss_scale string supports the following formats (segments separated by '+'):
    1. A strategy name alone (e.g., 'default', 'last_round', 'all') - uses base LossScale
    2. A loss scale type alone (e.g., 'hermes', 'react') - uses 'default' strategy
    3. A strategy name followed by a loss scale type (e.g., 'default+react', 'last_round+qwen')
    4. Multiple loss scale types chained together, optionally led by a base strategy
       (e.g., 'hermes+ignore_empty_think', 'last_round+hermes+ignore_empty_think').
       The chained loss scales are applied sequentially: each loss scale processes the
       output of the previous one and the corresponding weights are multiplied together.

    Args:
        loss_scale: String specifying the loss scale configuration.

    Returns:
        LossScale: An instance of the appropriate LossScale subclass. When multiple loss
            scale types are specified, a ``ConcatLossScale`` wrapping them is returned.

    Examples:
        >>> get_loss_scale('default')  # Uses default strategy with base LossScale
        >>> get_loss_scale('react')  # Uses default strategy with REACTLossScale
        >>> get_loss_scale('last_round+hermes')  # last_round strategy with HermesLossScale
        >>> get_loss_scale('last_round+hermes+ignore_empty_think')  # chain hermes then ignore_empty_think
    """
    parts = loss_scale.split('+')
    if parts[0] in ALL_BASE_STRATEGY:
        base_strategy = parts[0]
        ls_names = parts[1:] or ['base']
    else:
        base_strategy = 'default'
        ls_names = parts
    if len(ls_names) == 1:
        return loss_scale_map[ls_names[0]](base_strategy)
    # The base_strategy is owned by the outer ConcatLossScale; sub loss scales only
    # contribute their `get_loss_scale` (which does not reference base_strategy), so
    # any valid placeholder ('default') is fine here.
    sub_loss_scales = [loss_scale_map[name]('default') for name in ls_names]
    return ConcatLossScale(sub_loss_scales, base_strategy)

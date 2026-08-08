# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import ConfigLossScale


class IgnoreEmptyThinkLossScale(ConfigLossScale):
    loss_scale_config = 'ignore_empty_think.json'


class IgnoreThinkPrefixLossScale(ConfigLossScale):
    loss_scale_config = 'ignore_think_prefix.json'

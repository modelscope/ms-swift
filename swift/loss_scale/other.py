# Copyright (c) Alibaba, Inc. and its affiliates.
from .base import ConfigLossScale


class IgnoreEmptyThinkLossScale(ConfigLossScale):
    loss_scale_config = 'ignore_empty_think.json'

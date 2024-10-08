# Copyright (c) Alibaba, Inc. and its affiliates.
from .constant import TemplateType
from .register import register_template
from .base import Template


DEFAULT_SYSTEM = 'You are a helpful assistant.'

register_template(TemplateType.qwen, QwenTemplate())
register_template(TemplateType.qwen2_5, Qwen2_5Template())




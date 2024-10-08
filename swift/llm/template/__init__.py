# Copyright (c) Alibaba, Inc. and its affiliates.
from .base import Template
from .constant import TemplateType
from .register import TEMPLATE_MAPPING, get_template, register_template
from .utils import StopWords


def _register_files():
    from . import qwen
    from . import llama
    # TODO
    # from . import template


_register_files()

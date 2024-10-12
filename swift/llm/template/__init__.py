# Copyright (c) Alibaba, Inc. and its affiliates.
"""The directory will be migrated to the modelscope repository.
The `_utils.py` file will contain copies of functions related to swift,
allowing the directory to be independently runnable.

1. Copy the entire template directory to modelscope.
2. Delete the `base.py` file and rename `_base.py` to `base.py`.
"""

from .agent import get_tools_prompt, split_action_action_input
from .base import Template, TemplateInputs
from .constant import TemplateType
from .register import TEMPLATE_MAPPING, get_template, register_template
from .utils import Word


def _register_files():
    from . import qwen
    from . import llama
    # TODO
    # from . import template


_register_files()

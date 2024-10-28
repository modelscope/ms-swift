# Copyright (c) Alibaba, Inc. and its affiliates.
"""The directory will be migrated to the modelscope repository.
The `_utils.py` file will contain copies of functions related to swift,
allowing the directory to be independently runnable.
Please copy the entire template directory to modelscope.
"""

from . import template
from .agent import get_tools_prompt, split_action_action_input
from .base import Template, TemplateInputs
from .constant import TemplateType
from .register import TEMPLATE_MAPPING, get_template, register_template
from .utils import Word

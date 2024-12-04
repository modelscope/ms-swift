# Copyright (c) Alibaba, Inc. and its affiliates.
"""The directory will be migrated to the modelscope repository.
The `_utils.py` file will contain copies of functions related to swift,
allowing the directory to be independently runnable.
Please copy the entire template directory to modelscope.
"""

from . import template
from .base import Template
from .constant import TemplateType
from .register import TEMPLATE_MAPPING, get_template, get_template_meta, register_template
from .template_inputs import InferRequest, Messages, TemplateInputs, Tool
from .template_meta import TemplateMeta
from .utils import Word, split_action_action_input, split_parts_by_regex, split_str_parts_by

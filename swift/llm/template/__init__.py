# Copyright (c) Alibaba, Inc. and its affiliates.
from . import template
from .base import MaxLengthError, Template
from .constant import TemplateType
from .register import TEMPLATE_MAPPING, get_template, get_template_meta, register_template
from .template_inputs import InferRequest, TemplateInputs
from .template_meta import TemplateMeta
from .utils import Word, split_action_action_input, split_parts_by_regex, split_str_parts_by
from .vision_utils import load_file, load_image

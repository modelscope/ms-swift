# Copyright (c) Alibaba, Inc. and its affiliates.
"""The directory will be migrated to the modelscope repository.
The `_utils.py` file will contain copies of functions related to swift,
allowing the directory to be independently runnable.

1. Copy the entire template directory to modelscope.
2. Delete the `base.py` file and rename `_base.py` to `base.py`.
"""

from .base import Template
from .constant import TemplateType
from .register import TEMPLATE_MAPPING, get_template, register_template
from .utils import StopWords

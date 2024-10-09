# Copyright (c) Alibaba, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from .base import Template

TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {}


def register_template(template_type: str, template: Template, *, exist_ok: bool = False, **kwargs) -> None:
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    template.template_type = template_type
    template_info = {'template': template, **kwargs}
    TEMPLATE_MAPPING[template_type] = template_info


def get_template(
    template_type: str,
    tokenizer: PreTrainedTokenizerBase,
    default_system: Optional[str] = None,
) -> 'Template':
    template_info = TEMPLATE_MAPPING[template_type]
    # To ensure that obtaining the same template_type multiple times does not interfere with each other.
    template = deepcopy(template_info['template'])
    template.init_template(tokenizer, default_system)
    return template

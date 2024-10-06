# Copyright (c) Alibaba, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, Literal, Optional

from transformers import PreTrainedTokenizerBase

from ._template import Template as _Template

TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {}


class Template(_Template):
    pass


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
    max_length: Optional[int] = None,
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
    **kwargs,
) -> 'Template':
    template_info = TEMPLATE_MAPPING[template_type]
    template = deepcopy(template_info['template'])
    template.init_template(tokenizer, default_system, max_length, truncation_strategy, **kwargs)
    return template

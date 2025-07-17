# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, Literal, Optional

from ..utils import Processor
from .base import Template
from .template_meta import TemplateMeta

TEMPLATE_MAPPING: Dict[str, TemplateMeta] = {}


def register_template(template_meta: TemplateMeta, *, exist_ok: bool = False) -> None:
    template_type = template_meta.template_type
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    TEMPLATE_MAPPING[template_type] = template_meta


def get_template(
    template_type: str,
    processor: Processor,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    *,
    truncation_strategy: Literal['raise', 'left', 'right'] = 'raise',
    max_pixels: Optional[int] = None,  # h * w
    agent_template: Optional[str] = None,
    norm_bbox: Literal['norm1000', 'none', None] = None,
    use_chat_template: bool = True,
    remove_unused_columns: bool = True,
    # train
    padding_free: bool = False,
    padding_side: Literal['left', 'right'] = 'right',
    loss_scale: str = 'default',
    sequence_parallel_size: int = 1,
    # infer/deploy
    response_prefix: Optional[str] = None,
    template_backend: Literal['swift', 'jinja'] = 'swift',
) -> 'Template':
    template_meta = TEMPLATE_MAPPING[template_type]
    template_cls = template_meta.template_cls
    return template_cls(
        processor,
        template_meta,
        default_system,
        max_length,
        truncation_strategy=truncation_strategy,
        max_pixels=max_pixels,
        agent_template=agent_template,
        norm_bbox=norm_bbox,
        use_chat_template=use_chat_template,
        remove_unused_columns=remove_unused_columns,
        # train
        padding_free=padding_free,
        padding_side=padding_side,
        loss_scale=loss_scale,
        sequence_parallel_size=sequence_parallel_size,
        # infer/deploy
        response_prefix=response_prefix,
        template_backend=template_backend,
    )


def get_template_meta(template_type: str) -> TemplateMeta:
    return TEMPLATE_MAPPING[template_type]

# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, Literal, Optional

from transformers import AutoConfig, PretrainedConfig

from swift.utils import HfConfigFactory, Processor, safe_snapshot_download
from .base import Template
from .template_meta import TemplateMeta

TEMPLATE_MAPPING: Dict[str, TemplateMeta] = {}


def register_template(template_meta: TemplateMeta, *, exist_ok: bool = False) -> None:
    template_type = template_meta.template_type
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    TEMPLATE_MAPPING[template_type] = template_meta


def _read_args_json_template_type(model_dir):
    if not os.path.exists(os.path.join(model_dir, 'args.json')):
        return
    from swift.arguments import BaseArguments
    args = BaseArguments.from_pretrained(model_dir)
    return args.template_type


def get_template(
    processor: Processor,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    *,
    template_type: Optional[str] = None,
    truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise',
    max_pixels: Optional[int] = None,  # h * w
    agent_template: Optional[str] = None,
    norm_bbox: Literal['norm1000', 'none', None] = None,
    use_chat_template: bool = True,
    remove_unused_columns: bool = True,
    padding_side: Literal['left', 'right'] = 'right',
    # train
    padding_free: bool = False,
    loss_scale: str = 'default',
    sequence_parallel_size: int = 1,
    # infer/deploy
    template_backend: Literal['swift', 'jinja'] = 'swift',
    # thinking
    response_prefix: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    add_non_thinking_prefix: bool = True,
    # hub
    use_hf: Optional[bool] = None,
    hub_token: Optional[str] = None,
    revision: Optional[str] = None,
) -> 'Template':
    model_info = processor.model_info
    model_meta = processor.model_meta
    template_type = template_type or model_meta.template
    if template_type is None:
        template_type = _read_args_json_template_type(model_info.model_dir)
    if template_type is None:
        candidates = model_meta.candidate_templates
        if len(candidates) > 1 or len(candidates) == 0:
            candidates_str = ''
            if len(candidates) > 1:
                candidates_str = f'Multiple possible types found: {candidates}. '
            raise ValueError(
                f'Failed to automatically match `template_type` for `{model_info.model_dir}`. {candidates_str}'
                'Please specify `template_type` manually. See documentation: '
                'https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html')
        elif len(candidates) == 1:
            template_type = candidates[0]
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
        padding_side=padding_side,
        # train
        padding_free=padding_free,
        loss_scale=loss_scale,
        sequence_parallel_size=sequence_parallel_size,
        # infer/deploy
        template_backend=template_backend,
        # thinking
        response_prefix=response_prefix,
        enable_thinking=enable_thinking,
        add_non_thinking_prefix=add_non_thinking_prefix,
    )


def get_template_meta(template_type: str) -> TemplateMeta:
    return TEMPLATE_MAPPING[template_type]

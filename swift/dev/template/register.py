# Copyright (c) ModelScope Contributors. All rights reserved.

import os
from typing import Any, Dict, FrozenSet, Literal, Optional

import json

from .base import Template
from .meta import TemplateMeta
from .processor import Processor

TEMPLATE_MAPPING: Dict[str, TemplateMeta] = {}
_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    from . import families  # noqa: F401 -- family modules register their metadata on import


def register_template(template_meta: TemplateMeta, *, exist_ok: bool = False) -> None:
    template_type = template_meta.template_type
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f"The `{template_type}` has already been registered in the TEMPLATE_MAPPING.")
    TEMPLATE_MAPPING[template_type] = template_meta


def _read_args_json_template_type(model_dir):
    path = os.path.join(model_dir, "args.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file).get("template")


def get_template_meta(model_info: Any, model_meta: Any, template_type: Optional[str] = None) -> TemplateMeta:
    _ensure_registered()
    if template_type is None and model_info is not None:
        template_type = _read_args_json_template_type(model_info.model_dir)
    template_type = template_type or model_meta.template
    if template_type is None:
        candidates = model_meta.candidate_templates
        if len(candidates) > 1 or len(candidates) == 0:
            candidates_str = ""
            if len(candidates) > 1:
                candidates_str = f"Multiple possible types found: {candidates}. "
            raise ValueError(
                f"Failed to automatically match `template_type` for `{model_info.model_dir}`. {candidates_str}"
                "Please specify `template_type` manually via `--template`. See documentation: "
                "https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html"
            )
        elif len(candidates) == 1:
            template_type = candidates[0]
    elif template_type not in TEMPLATE_MAPPING:
        raise ValueError(f"template_type: '{template_type}' not in {list(TEMPLATE_MAPPING.keys())}")
    template_meta = TEMPLATE_MAPPING[template_type]
    return template_meta


def get_template(
    processor: Processor,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    *,
    template_type: Optional[str] = None,
    truncation_strategy: Literal["raise", "left", "right", "split"] = "raise",
    max_pixels: Optional[int] = None,  # h * w
    agent_template: Optional[str] = None,
    norm_bbox: Literal["norm1000", "none", None] = None,
    use_chat_template: bool = True,
    remove_unused_columns: bool = True,
    padding_side: Literal["left", "right"] = "right",
    # train
    padding_free: bool = False,
    loss_scale: str = "default",
    is_binary_loss_scale: Optional[bool] = None,
    sequence_parallel_size: int = 1,
    # infer/deploy
    template_backend: Literal["swift", "jinja"] = "swift",
    # thinking
    response_prefix: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    preserve_thinking: Optional[bool] = None,
    add_non_thinking_prefix: bool = True,
    media_types: FrozenSet[Literal["image", "video", "audio"]],
    task_type: Optional[str] = None,
    model_factory: Optional[Any] = None,
    multimodal_graph_adapter: Optional[Any] = None,
) -> "Template":
    """Resolve the registered family for this processor and instantiate it."""
    model_info = processor.model_info
    model_meta = processor.model_meta
    template_meta = get_template_meta(model_info, model_meta, template_type=template_type)
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
        is_binary_loss_scale=is_binary_loss_scale,
        sequence_parallel_size=sequence_parallel_size,
        # infer/deploy
        template_backend=template_backend,
        # thinking
        response_prefix=response_prefix,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
        add_non_thinking_prefix=add_non_thinking_prefix,
        media_types=media_types,
        task_type=task_type,
        model_factory=model_factory,
        multimodal_graph_adapter=multimodal_graph_adapter,
    )

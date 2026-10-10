# Copyright (c) ModelScope Contributors. All rights reserved.
"""Chat template, formatting, truncation, and loss configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TemplateConfig:
    """Template selection, sequence control, and thinking mode settings."""

    template: Optional[str] = None
    system: Optional[str] = None
    max_length: Optional[int] = None
    truncation_strategy: Literal['delete', 'left', 'right', 'split', None] = None
    max_pixels: Optional[int] = None
    norm_bbox: Literal['norm1000', 'none', None] = None
    use_chat_template: Optional[bool] = None
    padding_side: Literal['left', 'right'] = 'right'
    padding_free: bool = False
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1
    response_prefix: Optional[str] = None
    enable_thinking: Optional[bool] = None
    add_non_thinking_prefix: bool = True

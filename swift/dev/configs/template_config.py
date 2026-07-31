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
    agent_template: Optional[str] = None
    norm_bbox: Literal['norm1000', 'none', None] = None
    use_chat_template: Optional[bool] = None
    padding_side: Literal['left', 'right'] = 'right'
    padding_free: bool = False
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1
    is_binary_loss_scale: Optional[bool] = None
    template_backend: Literal['swift', 'jinja'] = 'swift'
    response_prefix: Optional[str] = None
    enable_thinking: Optional[bool] = None
    preserve_thinking: Optional[bool] = None
    add_non_thinking_prefix: bool = True
    disable_ignore_empty_think: bool = False

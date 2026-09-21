"""Gradio application configuration."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class AppConfig:
    """UI address, presentation, and remote deployment selection."""

    base_url: Optional[str] = None
    studio_title: Optional[str] = None
    is_multimodal: Optional[bool] = None
    server_name: str = '0.0.0.0'
    server_port: int = 7860
    share: bool = False
    lang: Literal['en', 'zh'] = 'en'

"""Logging and experiment tracking configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# TODO: integrate it
@dataclass
class LoggingConfig:
    """SwanLab, TensorBoard, and general logging settings."""

    # === SwanLab ===
    swanlab_token: Optional[str] = None
    swanlab_project: str = 'ms-swift'
    swanlab_workspace: Optional[str] = None
    swanlab_exp_name: Optional[str] = None
    swanlab_notification_method: Optional[str] = None
    swanlab_webhook_url: Optional[str] = None
    swanlab_secret: Optional[str] = None
    swanlab_mode: Literal['cloud', 'local'] = 'cloud'

    # === General Logging ===
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    logging_dir: Optional[str] = None
    logging_steps: int = 5
    logging_first_step: bool = True

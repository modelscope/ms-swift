"""Checkpoint saving and resumption configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class CheckpointConfig:
    """Model saving, resumption, and checkpoint management settings."""

    # === Save Strategy ===
    output_dir: str = 'output'
    save_strategy: Literal['steps', 'epoch', 'no', 'best'] = 'steps'
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    safe_serialization: bool = True
    max_shard_size: str = '5GB'
    save_on_each_node: bool = False
    save_only_model: bool = False

    # === Resume ===
    resume_from_checkpoint: Optional[str] = None
    resume_only_model: bool = False
    ignore_data_skip: bool = False

    # === Misc ===
    add_version: bool = True
    create_checkpoint_symlink: bool = False
    use_flash_ckpt: bool = False
    load_args: bool = False
    load_data_args: bool = False

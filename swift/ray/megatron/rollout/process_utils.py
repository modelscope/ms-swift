# Copyright (c) ModelScope Contributors. All rights reserved.
"""Process and CUDA utility functions for rollout actors."""
from __future__ import annotations

import torch


def set_expandable_segments(enable: bool) -> None:
    """Toggle CUDA expandable segments allocator setting (no-op on NPU)."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.memory._set_allocator_settings(f'expandable_segments:{enable}')
    except Exception:
        pass

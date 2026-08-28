# Copyright (c) ModelScope Contributors. All rights reserved.
"""Torch data-movement helpers, copied from ``swift.utils.torch_utils``."""
from collections.abc import Mapping
from typing import Any, Union

import torch


def to_device(data: Any, device: Union[str, torch.device, int], non_blocking: bool = False) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device, non_blocking) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        return data

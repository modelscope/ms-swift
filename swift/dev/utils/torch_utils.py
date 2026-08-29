# Copyright (c) ModelScope Contributors. All rights reserved.
"""Torch data-movement helpers, copied from ``swift.utils.torch_utils``."""
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, List, Tuple, Union

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


def get_n_params_grads(model) -> Tuple[List[int], List[int]]:
    """Per-parameter element counts and trainable-element counts (copied from ``swift.utils``).

    Under DeepSpeed ZeRO-3 each parameter is sharded, so it is gathered inside
    ``GatheredParameters`` before counting; otherwise counting is direct.
    """
    from transformers.integrations import is_deepspeed_zero3_enabled
    n_params, n_grads = [], []
    for p in model.parameters():
        if is_deepspeed_zero3_enabled():
            import deepspeed
            context = deepspeed.zero.GatheredParameters(p)
        else:
            context = nullcontext()
        with context:
            n_params.append(p.numel())
            n_grads.append(p.numel() if p.requires_grad else 0)
    return n_params, n_grads

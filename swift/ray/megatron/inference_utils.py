# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from typing import Any, Dict, List


def _slice_value_for_microbatch(value: Any, start: int, end: int, micro_batch_size: int) -> Any:
    """Recursively slice a value for one micro-batch."""
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] > micro_batch_size:
        return value[start:end]
    if isinstance(value, list) and len(value) > micro_batch_size:
        return value[start:end]
    if isinstance(value, tuple) and len(value) > micro_batch_size:
        return value[start:end]
    if isinstance(value, dict):
        return {k: _slice_value_for_microbatch(v, start, end, micro_batch_size) for k, v in value.items()}
    return value


def slice_batch(batch: Dict[str, Any], start: int, end: int, micro_batch_size: int) -> Dict[str, Any]:
    """Slice one micro-batch from a batch dict."""
    return {k: _slice_value_for_microbatch(v, start, end, micro_batch_size) for k, v in batch.items()}


def split_micro_batches(batch, micro_batch_size: int) -> List[Dict[str, Any]]:
    """Split a batch dict into micro-batch dicts along dim-0."""
    if isinstance(batch, list):
        return batch

    t = batch.get('input_ids')
    if not (isinstance(t, torch.Tensor) and t.dim() > 0):
        t = next((v for v in batch.values() if isinstance(v, torch.Tensor) and v.dim() > 0), None)
    sample_size = t.shape[0] if t is not None else int(batch.get('num_samples', 1))

    if micro_batch_size >= sample_size:
        return [batch]
    return [
        slice_batch(batch, i, min(i + micro_batch_size, sample_size), micro_batch_size)
        for i in range(0, sample_size, micro_batch_size)
    ]

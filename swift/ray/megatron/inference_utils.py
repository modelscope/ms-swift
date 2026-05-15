# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from typing import Any, Dict, List


def slice_batch(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Slice a batch along dim 0 for micro-batching."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] >= end:
            result[k] = v[start:end]
        elif isinstance(v, list) and len(v) >= end:
            result[k] = v[start:end]
        else:
            result[k] = v
    return result


def split_micro_batches(batch, micro_batch_size: int) -> List[Dict[str, Any]]:
    """Split a batch dict into micro-batch dicts along the batch dimension.

    Infers actual shard size from the first tensor dimension rather than
    ``num_samples`` (which may still reflect the global batch after DP dispatch).
    """
    if isinstance(batch, list):
        return batch
    t = next((v for v in batch.values() if isinstance(v, torch.Tensor) and v.dim() > 0), None)
    n = t.shape[0] if t is not None else int(batch.get('num_samples', 1))
    batch['num_samples'] = n
    if micro_batch_size >= n:
        return [batch]
    return [slice_batch(batch, i, min(i + micro_batch_size, n)) for i in range(0, n, micro_batch_size)]

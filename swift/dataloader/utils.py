# Copyright (c) ModelScope Contributors. All rights reserved.
import torch


def shuffle_batch_blocks(indices, batch_size, generator):
    """Shuffle complete DP micro-batches without changing their contents or rank assignments.

    Keep the first block for early OOM detection and leave an incomplete tail at the end.
    Call before rank sharding and before skipping consumed samples when resuming.
    """
    num_batches = len(indices) // batch_size
    if num_batches <= 2:
        return indices
    order = (torch.randperm(num_batches - 1, generator=generator) + 1).tolist()
    result = indices[:batch_size]
    for i in order:
        result.extend(indices[i * batch_size:(i + 1) * batch_size])
    result.extend(indices[num_batches * batch_size:])
    return result

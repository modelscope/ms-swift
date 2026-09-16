# Copyright (c) ModelScope Contributors. All rights reserved.
"""Miscellaneous utilities shared across the expert_parallel package."""
import torch

from swift.utils import get_logger

logger = get_logger()


def cast_trainable_params_to_uniform_dtype(model, target_dtype=torch.bfloat16):
    """Ensure all trainable parameters have the same dtype.

    FSDP2 requires uniform dtype across all trainable parameters.  Some
    models (e.g. DeepSeek-V4) store a subset of weights as ``float32``
    (HyperConnection params, RMSNorm, attention sinks, etc.) while the
    rest of the model is ``bfloat16``, causing FSDP2 to fail with::

        AssertionError: FSDP expects uniform original parameter dtype
            but got {torch.bfloat16, torch.float32}

    This function casts any stray ``float32`` (or other) trainable
    parameters to *target_dtype* so that FSDP2 wrapping succeeds.
    """
    trainable_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    if len(trainable_dtypes) <= 1:
        return
    logger.info(f'FSDP2 requires uniform parameter dtype, but found mixed dtypes: {trainable_dtypes}. '
                f'Casting all trainable parameters to {target_dtype}.')
    for p in model.parameters():
        if p.requires_grad and p.dtype != target_dtype:
            p.data = p.data.to(target_dtype)
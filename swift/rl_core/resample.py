# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared resample logic for HF / Megatron / Megatron-Ray trainers.

When ``truncation_strategy='delete'`` (or dynamic sampling), samples whose
``template.encode`` fails (e.g. exceed ``max_length``, or multimodal processing
errors) must be replaced with fresh ones drawn from a backup iterator until we
have ``len(inputs)`` valid samples. The backends previously each carried a
near-identical copy of this loop; they only differed in the iterator they pull
from and whether the assistant response is stripped before encoding (prompt-only
algorithms like GRPO strip; GKD off-policy distillation keeps it).
"""
from typing import Iterator, List

from swift.template.base import Template
from swift.utils import get_logger, remove_response

logger = get_logger()


def resample_encode_failed_inputs(
    template: Template,
    data_iterator: Iterator,
    inputs: List[dict],
    max_resample_rounds: int = 10,
    strip_response: bool = True,
) -> List[dict]:
    """Replace samples whose encode fails with fresh ones from ``data_iterator``.

    Caps the TOTAL encode attempts (fail-fast): a systematic failure
    (e.g. ``max_length`` too small so every prompt is over-length) raises quickly
    instead of churning through the iterator, and an empty batch breaks the loop
    instead of spinning forever.

    Args:
        template: Template used to encode (and thereby validate) a sample.
        data_iterator: Backup iterator yielding batches (lists) of fresh samples.
        inputs: The current batch; its length is the required valid count.
        max_resample_rounds: Resample budget; total attempts == required * (rounds + 1).
        strip_response: Remove the assistant response (in place) before encoding.

    Returns:
        A list of valid samples with the same length as ``inputs``.

    Raises:
        RuntimeError: If not enough valid samples are collected after the budget.
    """
    required = len(inputs)
    max_attempts = required * (max_resample_rounds + 1)
    valid, pending = [], list(inputs)
    attempts = n_dropped = 0

    while len(valid) < required and attempts < max_attempts:
        if not pending:
            batch = list(next(data_iterator))
            if not batch:  # guard: an empty batch would otherwise spin forever
                break
            pending.extend(batch)
        item = pending.pop(0)
        attempts += 1
        try:
            if strip_response:
                remove_response(item['messages'])
            template.encode(item)
            valid.append(item)
        except Exception as e:
            n_dropped += 1
            logger.info(f'Encoding failed for one sample; will resample. {e}')

    if len(valid) < required:
        raise RuntimeError(f'resample: only collected {len(valid)}/{required} valid samples after {attempts} '
                           f'attempts ({n_dropped} failed). Increase `max_length` or adjust `truncation_strategy`.')
    return valid[:required]

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dataset progress tracking collator wrapper.

This module provides a wrapper collator that extracts dataset source information
for progress tracking during training.
"""
from typing import Any, Callable, Dict, List


class ProgressTrackingCollator:
    """Wrapper collator that extracts dataset sources for progress tracking.

    This wrapper intercepts the collator output, extracts _dataset_source field
    from each sample, and passes it through to the main process via _batch_sources
    field for statistics collection.

    This approach is non-invasive - it doesn't modify any template code, only
    wraps the collator at the training level.

    Args:
        collator: The original collator function to wrap.
        track_progress: Whether to track progress. If False, just removes
            _dataset_source field without collecting statistics.

    Example:
        >>> original_collator = partial(template.data_collator, padding_to=None)
        >>> wrapped = ProgressTrackingCollator(original_collator)
        >>> batch_result = wrapped(batch)  # Contains _batch_sources field
    """

    def __init__(self, collator: Callable, track_progress: bool = True):
        self.collator = collator
        self.track_progress = track_progress

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process batch and extract dataset sources.

        Args:
            batch: List of encoded samples, each may contain _dataset_source field.

        Returns:
            Collated batch dict with optional _batch_sources field for progress tracking.
        """
        # 1. Collect sources before calling original collator
        # (original collator may modify batch in place)
        batch_sources = []
        if self.track_progress:
            for b in batch:
                sources = b.pop('_dataset_source', None)
                if sources is None:
                    continue
                # Handle both single value (non-packing) and list (packing)
                if isinstance(sources, str):
                    sources = [sources]
                for s in sources:
                    if s is not None:
                        batch_sources.append(s)
        else:
            # Not tracking, just remove the field to avoid issues
            for b in batch:
                b.pop('_dataset_source', None)

        # 2. Call original collator
        result = self.collator(batch)

        # 3. Attach sources for main process to collect
        if batch_sources:
            result['_batch_sources'] = batch_sources

        return result

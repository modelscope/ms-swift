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
        
        def _extract_source(item):
            if isinstance(item, dict):
                sources = item.get('_dataset_source')
                if sources is not None:
                    del item['_dataset_source']
                    return sources
            return None

        for b in batch:
            # Handle Packing scenario where batch item is a list of samples
            if isinstance(b, list):
                for sub_item in b:
                    sources = _extract_source(sub_item)
                    if self.track_progress and sources:
                        if isinstance(sources, str):
                            batch_sources.append(sources)
                        elif isinstance(sources, list):
                            batch_sources.extend(sources)
            # Handle normal scenario where batch item is a single sample dict
            elif isinstance(b, dict):
                sources = _extract_source(b)
                if self.track_progress and sources:
                     if isinstance(sources, str):
                        batch_sources.append(sources)
                     elif isinstance(sources, list):
                        batch_sources.extend(sources)

        # 2. Call original collator
        result = self.collator(batch)

        # 3. Attach sources for main process to collect
        if batch_sources:
            result['_batch_sources'] = batch_sources

        return result

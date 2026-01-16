# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dataset progress tracking collator wrapper.

This module provides a wrapper collator that extracts dataset source information
for progress tracking during training.
"""
from typing import Any, Callable, Dict, List


class ProgressTrackingCollator:
    """Wrapper collator that extracts dataset sources and token lengths for progress tracking.

    This wrapper intercepts the collator output, extracts _dataset_source and length fields
    from each sample, and passes them through to the main process via _batch_sources and
    _batch_lengths fields for statistics collection.

    This approach is non-invasive - it doesn't modify any template code, only
    wraps the collator at the training level.

    Args:
        collator: The original collator function to wrap.
        track_progress: Whether to track progress. If False, just removes
            _dataset_source field without collecting statistics.

    Example:
        >>> original_collator = partial(template.data_collator, padding_to=None)
        >>> wrapped = ProgressTrackingCollator(original_collator)
        >>> batch_result = wrapped(batch)  # Contains _batch_sources and _batch_lengths fields
    """

    def __init__(self, collator: Callable, track_progress: bool = True):
        self.collator = collator
        self.track_progress = track_progress

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process batch and extract dataset sources and token lengths.

        Args:
            batch: List of encoded samples, each may contain _dataset_source and length fields.

        Returns:
            Collated batch dict with optional _batch_sources and _batch_lengths fields.
        """
        # 1. Collect sources and lengths before calling original collator
        # (original collator may modify batch in place)
        batch_sources = []
        batch_lengths = []
        
        def _extract_info(item):
            if isinstance(item, dict):
                sources = item.get('_dataset_source')
                if sources is not None:
                    del item['_dataset_source']
                # Extract length but don't delete it (may be needed later)
                length = item.get('length')
                return sources, length
            return None, None

        for b in batch:
            # Handle Packing scenario where batch item is a list of samples
            if isinstance(b, list):
                for sub_item in b:
                    sources, length = _extract_info(sub_item)
                    if self.track_progress and sources:
                        if isinstance(sources, str):
                            batch_sources.append(sources)
                        elif isinstance(sources, list):
                            batch_sources.extend(sources)
                    if length is not None:
                        batch_lengths.append(length)
            # Handle normal scenario where batch item is a single sample dict
            elif isinstance(b, dict):
                sources, length = _extract_info(b)
                if self.track_progress and sources:
                     if isinstance(sources, str):
                        batch_sources.append(sources)
                     elif isinstance(sources, list):
                        batch_sources.extend(sources)
                if length is not None:
                    batch_lengths.append(length)

        # 2. Call original collator
        result = self.collator(batch)

        # 3. Attach sources and lengths for main process to collect
        if batch_sources:
            result['_batch_sources'] = batch_sources
        if batch_lengths:
            result['_batch_lengths'] = batch_lengths

        return result

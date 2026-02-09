# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dataset progress tracking collator wrapper.

This module provides a wrapper collator that extracts dataset source information
for progress tracking during training.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple


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

    def _extract_info(self, item: Any) -> Tuple[Optional[Any], Optional[int]]:
        """Extract and remove _dataset_source, extract length from item."""
        if isinstance(item, dict):
            sources = item.pop('_dataset_source', None)
            length = item.get('length')
            return sources, length
        return None, None

    def _collect_sources_and_lengths(
        self,
        sources: Optional[Any],
        length: Optional[int],
        batch_sources: List[str],
        batch_lengths: List[int],
    ) -> None:
        """Collect sources and lengths into batch lists."""
        if self.track_progress and sources:
            if isinstance(sources, str):
                batch_sources.append(sources)
            elif isinstance(sources, list):
                batch_sources.extend(sources)
        if length is not None:
            batch_lengths.append(length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process batch and extract dataset sources and token lengths.

        Args:
            batch: List of encoded samples, each may contain _dataset_source and length fields.

        Returns:
            Collated batch dict with optional _batch_sources and _batch_lengths fields.
        """
        # 1. Collect sources and lengths before calling original collator
        # (original collator may modify batch in place)
        batch_sources: List[str] = []
        batch_lengths: List[int] = []

        for b in batch:
            # Handle both Packing scenario (list) and normal scenario (dict)
            items = b if isinstance(b, list) else [b]
            for item in items:
                sources, length = self._extract_info(item)
                self._collect_sources_and_lengths(sources, length, batch_sources, batch_lengths)

        # 2. Call original collator
        result = self.collator(batch)

        # 3. Attach sources and lengths for main process to collect
        if batch_sources:
            result['_batch_sources'] = batch_sources
        if batch_lengths:
            result['_batch_lengths'] = batch_lengths

        return result

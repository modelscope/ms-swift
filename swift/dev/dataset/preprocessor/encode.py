# Copyright (c) ModelScope Contributors. All rights reserved.
"""An eager preprocessing pass that runs ``template.encode`` on every row via HF ``map``."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import Preprocessor

if TYPE_CHECKING:
    from swift.template import Template

__all__ = ['EncodePreprocessor']


class EncodePreprocessor(Preprocessor):
    """Run ``template.encode`` on each row, producing ``input_ids`` / ``labels`` / ``attention_mask``.

    The pass that materialises an encoding, as opposed to :class:`~swift.dev.dataset.EncodedDataset`,
    which is the view over what this leaves behind. Needed where a row cannot be encoded on access:
    ``truncation_strategy='split'`` turns one long text into several samples, so the row count itself
    depends on encoding and has to be settled before anything counts steps or plans packs. A stream has
    no separate pass to put encoding in either, so it runs this one inline.
    """

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row, return_length=True)

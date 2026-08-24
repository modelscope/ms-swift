# Copyright (c) ModelScope Contributors. All rights reserved.
"""An eager preprocessing pass that runs ``template.encode`` on every row via HF ``map``."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional

from .preprocessor import Preprocessor

if TYPE_CHECKING:
    from swift.template import Template

__all__ = ['EncodePreprocessor']


class EncodePreprocessor(Preprocessor):
    """Run ``template.encode`` on each row, producing ``input_ids`` / ``labels`` / ``attention_mask``.

    Used in the ``split`` training mode, where one text row fans out into several samples of varying
    position, and therefore must be fully encoded before packing can decide how long each piece is.
    """

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row, return_length=True)

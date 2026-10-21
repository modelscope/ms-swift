# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import Preprocessor

if TYPE_CHECKING:
    from swift.template import Template

__all__ = ['EncodePreprocessor']


class EncodePreprocessor(Preprocessor):
    """Run ``template.encode`` on each row, producing ``input_ids`` / ``labels`` / ``attention_mask``.

    The materialising pass, as opposed to ``EncodedDataset`` (the view over what it leaves behind):
    needed where the row count itself depends on encoding (``truncation_strategy='split'``), or on a
    stream, which has no separate pass to put encoding in.
    """

    def __init__(self, template: 'Template', *, seed: int = 42):
        super().__init__(seed=seed)
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row, return_length=True)

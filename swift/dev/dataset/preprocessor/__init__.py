# Copyright (c) ModelScope Contributors. All rights reserved.
"""The row-transform execution layer: map a per-row transform over a dataset, dropping bad rows.

:class:`Preprocessor` owns the ``map`` call itself. The two subclasses here are the passes that run
``template.encode`` over a whole dataset: :class:`EncodePreprocessor` keeps what it produces, and
:class:`MeasurePreprocessor` keeps only the token count it reveals.
"""
from .base import MessagesRepairPreprocessor, Preprocessor
from .encode import EncodePreprocessor
from .measure import MeasurePreprocessor

__all__ = ['MessagesRepairPreprocessor', 'Preprocessor', 'EncodePreprocessor', 'MeasurePreprocessor']

# Copyright (c) ModelScope Contributors. All rights reserved.
# Weight transforms: the model -> model components (merge / quantize / convert).
from __future__ import annotations

from .base import BaseWeightTransform
from .convert import ConvertComponent
from .merge import MergeComponent
from .quantize import QuantizeComponent

__all__ = ['BaseWeightTransform', 'ConvertComponent', 'MergeComponent', 'QuantizeComponent']

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight transforms: the model -> model components (merge / quantize / convert).

Every member loads a model directory, rewrites its weights, and writes a new directory out, so they
share ``BaseWeightTransform``'s validate -> resolve_output_dir -> skip -> run skeleton; each module
here adds one transform.
"""
from __future__ import annotations

from .base import BaseWeightTransform
from .convert import ConvertComponent
from .merge import MergeComponent
from .quantize import QuantizeComponent

__all__ = ['BaseWeightTransform', 'ConvertComponent', 'MergeComponent', 'QuantizeComponent']

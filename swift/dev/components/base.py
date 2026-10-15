# Copyright (c) ModelScope Contributors. All rights reserved.
"""The minimal component protocol.

A component is a class whose ``__init__`` binds configuration and whose ``__call__`` consumes the
upstream artifacts and returns the downstream ones. That is the whole contract: no shared loading
machinery, no registry. A recipe reads as a linear composition of these components, each fed the
artifact the previous one produced.
"""
from __future__ import annotations

from typing import Any


class RecipeComponent:
    """Base for every dev component.

    Subclasses implement ``__call__``. Nothing else is inherited on purpose -- components that share
    real work share a narrower base (see ``weight_transform.BaseWeightTransform``), not this one.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any


class RecipeComponent:
    """Base for every dev component: ``__init__`` binds config, ``__call__`` does the work.

    Nothing else is inherited on purpose -- components that share real work share a narrower base
    (see ``weight_transform.BaseWeightTransform``), not this one.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

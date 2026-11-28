# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any


class Recipe:
    """Base for every non-training recipe: ``__init__`` binds config, ``__call__`` does the work.

    Nothing else is inherited on purpose -- recipes that share real work share a narrower base
    (see ``export.base.BaseWeightTransform``), not this one.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

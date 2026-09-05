# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rule-based outcome reward models (ORMs) for swift.dev -- the built-in ``reward`` plugins.

Internalized from ``swift.rewards`` so swift.dev carries no runtime dependency on legacy swift
packages. ``ORM`` / ``AsyncORM`` are the historical names of swift's reward plugin bases (see
:mod:`swift.dev.plugin`); ``orms`` is the registry the ``reward`` extension point adopts, and
:data:`REWARD` is that point. Consumed by :mod:`swift.dev.reward`.
"""
from .orm import ORM, REWARD, AsyncORM, orms

__all__ = ['ORM', 'AsyncORM', 'REWARD', 'orms']

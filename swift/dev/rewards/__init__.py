# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rule-based outcome reward models (ORMs) for swift.dev.

Internalized from ``swift.rewards`` so swift.dev carries no runtime dependency on legacy swift
packages. The ``orms`` registry (accuracy / format / cosine / repetition / soft_overlong / ...) is
consumed by :mod:`swift.dev.reward`.
"""
from .orm import ORM, AsyncORM, orms

__all__ = ['ORM', 'AsyncORM', 'orms']

"""dev's minimal adapter over swift's Template: the twinkle contract only (`DevMixin`).

`build_template` derives `Shifted<LegacyClass>` via `shifted_template_class`, so the legacy class --
and every method its family overrode (encode/collate/media) -- stays in place and only the twinkle
contract is added on top. There is a single encode implementation (swift's) for both the dataset and
the model; this package is development-only scaffolding and disappears once `DevMixin` moves into
swift's Template proper.
"""
from __future__ import annotations

from .template import DevMixin, shifted_template_class

__all__ = ['DevMixin', 'shifted_template_class']

"""dev's label convention (`NextTokenShiftMixin`) plus the opt-in encode rewrite (`Template`).

Default path: `shifted_template_class(type(legacy))` mixes the shift into the legacy class, keeping
its own encode/collate/media behaviour. Opt-in path (TemplateConfig.legacy_encode=False): `Template`
and the per-family subclasses selected through `PROCESSOR_TEMPLATE_MAPPING` (keyed by legacy
template_type), not imported by name, so only the base class and the mapping are re-exported here.
"""
from __future__ import annotations

from .template import PROCESSOR_TEMPLATE_MAPPING, NextTokenShiftMixin, Template, shifted_template_class

__all__ = ['Template', 'NextTokenShiftMixin', 'shifted_template_class', 'PROCESSOR_TEMPLATE_MAPPING']

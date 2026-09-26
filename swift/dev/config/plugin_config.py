"""Plugin / extension-file configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PluginConfig:
    """User ``.py`` files imported before anything is built, so their ``@register`` calls run.

    These fields are cross-cutting -- one plugin file may register a model, a template, a dataset, a
    reward or a loss -- so they live in their own Config rather than riding on ``ModelConfig`` (which
    only happens to be the one Config every command parses). ``PluginRegistry.load_configured`` reads
    both fields and imports the files once, in the order given.
    """

    #: Python files imported before anything is built, so that decorated models, templates, datasets and
    #: reward functions register themselves. Import order is the order given.
    external_plugins: List[str] = field(default_factory=list)
    #: Files whose ``register_model`` / ``register_template`` calls add entries the built-in registries
    #: do not have. Distinct from ``external_plugins``, which is for behaviour rather than registration.
    #: Kept as a separate flag for swift 3.x compatibility; both lists are imported together.
    custom_register_path: List[str] = field(default_factory=list)

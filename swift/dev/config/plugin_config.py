"""Plugin / extension-file configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PluginConfig:
    """User plugin sources imported before anything is built, so their ``@register`` calls run.

    These fields are cross-cutting -- one plugin source may register a model, a template, a dataset, a
    reward or a loss -- so they live in their own Config rather than riding on ``ModelConfig`` (which
    only happens to be the one Config every command parses). ``PluginRegistry.load_configured`` reads
    the field and imports the sources once, in the order given.
    """

    #: Plugin sources imported before anything is built, so that decorated models, templates, datasets
    #: and reward functions register themselves. Each entry may be a local ``.py`` file, a local folder
    #: (its ``__init__.py``), or a ``hf://`` / ``ms://`` id that is downloaded if absent. Import order is
    #: the order given. This is the single loading entry point -- the legacy ``custom_register_path``
    #: (a second flag meaning the same "import this first") is gone; use ``external_plugins`` for both
    #: behaviour plugins and plain registrations.
    external_plugins: List[str] = field(default_factory=list)

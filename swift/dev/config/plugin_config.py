# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import List


@dataclass
class PluginConfig:
    """User plugin sources imported before anything is built, so their ``@register`` calls run.

    Cross-cutting (one source may register a model, template, dataset, reward or loss).
    """

    # A local .py file, a local folder, or a hf:// / ms:// id. The legacy custom_register_path is
    # gone; use external_plugins for both behaviour plugins and plain registrations.
    external_plugins: List[str] = field(default_factory=list)

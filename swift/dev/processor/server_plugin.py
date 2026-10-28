# Copyright (c) ModelScope Contributors. All rights reserved.
"""Server-side dev InputProcessor, registered into ``twinkle.processor`` at worker startup."""
from __future__ import annotations

from typing import Any, Dict

from swift.dev.processor.base import InputProcessor, processor_kwargs


class DevInputProcessor(InputProcessor):
    """A dev InputProcessor whose Template is rebuilt from a serialized ``TemplateSpec``.

    ``template_spec`` arrives as JSON in ``set_processor`` init_kwargs; the Template is rebuilt here
    so the per-row stages (position_ids/mrope, bookkeeping drops) run on the server, next to the
    batch stages twinkle already owns.
    """

    def __init__(self, *, template_spec: Dict[str, Any], **kwargs: Any) -> None:
        from swift.dev.builders.template import build_template_from_spec
        from swift.dev.template_contract.types import TemplateSpec

        template = build_template_from_spec(TemplateSpec.from_dict(template_spec))
        super().__init__(**{**processor_kwargs(template), **kwargs})


def register() -> None:
    """Export :class:`DevInputProcessor` into ``twinkle.processor`` so ``set_processor`` resolves it by name."""
    import twinkle.processor

    twinkle.processor.DevInputProcessor = DevInputProcessor

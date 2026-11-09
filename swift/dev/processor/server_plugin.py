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
    """Export dev classes into the twinkle namespaces the server resolves names through.

    ``DevInputProcessor`` goes into ``twinkle.processor`` (set_processor by name). The decision
    loss family goes into ``twinkle.loss`` via ``register_loss``'s export: the remote training
    path sets a task's loss by CLASS NAME, and the decision losses live in swift.dev -- importing
    the module here runs the registration that makes those names resolvable server-side.
    """
    import swift.dev.loss.decision  # noqa: F401 -- register_loss exports the classes into twinkle.loss
    import twinkle.processor

    twinkle.processor.DevInputProcessor = DevInputProcessor

# Copyright (c) ModelScope Contributors. All rights reserved.
"""An EXTERNAL sampler plugin: bring your own generation engine.

``--sampler`` normally names a built-in engine (``vllm`` / ``sglang`` / ``transformers``, with ``pt`` an
alias of ``transformers``). It also accepts an external SOURCE -- a local ``.py`` file, a local folder, or
a ``hf://`` / ``ms://`` id -- which ``swift.dev.builders.sampler._resolve_sampler`` resolves through the
same unified loader (``resolve_plugin_class`` -> ``Plugin.load_plugin``) and picks the twinkle ``Sampler``
subclass out of. So a custom sampler is selected by pointing ``--sampler`` AT THIS FILE::

    swift infer ... --sampler examples/v5/infer/custom_sampler.py

Why a SEPARATE file from ``custom_plugins.py`` (which carries the other five kinds): the 'sampler' plugin
kind is declared LAZILY, inside ``build_sampler``, because importing ``twinkle.sampler`` pulls in vLLM and
torch (~10s). A sampler is therefore resolved by source at build time, NOT registered by an ``@register``
that runs at ``--external_plugins`` import time -- and keeping the heavy ``TransformersSampler`` import out
of the shared plugin file means a training run that builds no sampler never pays for it.

Constructor contract (what ``build_sampler`` calls, and so what any custom sampler must accept)::

    Cls(model, *, engine_args, device_mesh, remote_group)

``engine_args`` is passed VERBATIM for a custom sampler -- dev applies none of the built-in engines'
ModelConfig/LoRA/pooling knob mapping, so this class owns its own knob vocabulary. ``device_mesh`` carries
data parallelism; ``remote_group`` names the twinkle DeviceGroup to place the engine in under ``mode='ray'``
(omitted in local mode). ``build_sampler`` then calls ``set_template(template)`` on the instance.
"""
from __future__ import annotations

import logging
from typing import Any, List

from twinkle.sampler import TransformersSampler

logger = logging.getLogger(__name__)


class LoggingTransformersSampler(TransformersSampler):
    """The transformers engine with one extra log line per ``sample`` call.

    A real custom sampler would replace the engine outright (a different inference stack, a mocked
    generator for tests, a wrapper that adds caching or routing). This one keeps the whole transformers
    contract by inheriting it and only interposing on ``sample``, which is the smallest change that still
    proves the file was resolved and constructed as the run's sampler. ``__init__`` is inherited verbatim,
    so the ``Cls(model, *, engine_args, device_mesh, remote_group)`` contract above is met by the base.
    """

    def sample(self, trajectories: Any, *args, **kwargs) -> List[Any]:
        count = len(trajectories) if hasattr(trajectories, '__len__') else 1
        logger.info(f'[custom_sampler] LoggingTransformersSampler sampling {count} trajector(y/ies)')
        return super().sample(trajectories, *args, **kwargs)

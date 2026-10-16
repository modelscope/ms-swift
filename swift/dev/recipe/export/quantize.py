# Copyright (c) ModelScope Contributors. All rights reserved.
"""The ``run_quantize`` entry point.

Quantization itself lives in :class:`swift.dev.components.QuantizeComponent`. This module keeps the
procedural signature the export CLI calls and returns a plain ``str``, so callers are unaffected by
the component's typed artifact.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from swift.dev.components import QuantizeComponent

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, ModelConfig, QuantizeConfig, TemplateConfig


def run_quantize(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    quantize_config: QuantizeConfig,
    dataset_config: Optional[DatasetConfig] = None,
    *,
    output_dir: str = 'output',
    quant_n_samples: int = 256,
    safe_serialization: bool = True,
    max_shard_size: str = '5GB',
    **quantizer_kwargs,
) -> str:
    """Quantize ``model_config`` and write the result to ``output_dir``; returns that path.

    ``quantizer_kwargs`` are forwarded verbatim to the twinkle quantizer, which is where every
    backend knob already lives (group_size / zero_point / version / v2 / bnb_4bit_* ...). They are
    NOT mirrored onto QuantizeConfig: that would duplicate the same fields in two places and let
    them drift.

    ``dataset_config`` is required for AWQ/GPTQ and unused otherwise. ``quant_n_samples`` caps how
    many rows are encoded for calibration (legacy's ``--quant_n_samples``).
    """
    component = QuantizeComponent(
        model_config,
        template_config,
        quantize_config,
        dataset_config,
        output_dir=output_dir,
        quant_n_samples=quant_n_samples,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
        **quantizer_kwargs)
    return component().path

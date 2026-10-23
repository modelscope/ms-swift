# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, ModelConfig, QuantizeConfig, TemplateConfig


def run_quantize(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    quantize_config: QuantizeConfig,
    dataset_config: Optional[DatasetConfig] = None,
    *,
    output_dir: str = 'output',
    quant_n_samples: Optional[int] = None,
    safe_serialization: bool = True,
    max_shard_size: str = '5GB',
    exist_ok: bool = False,
    **quantizer_kwargs,
) -> str:
    """Quantize ``model_config`` and write the result to ``output_dir``; returns that path.

    ``quantizer_kwargs`` go verbatim to the twinkle quantizer, where every backend knob lives --
    mirroring them onto QuantizeConfig would duplicate fields and let them drift.
    ``dataset_config`` feeds AWQ/GPTQ calibration; ``quant_n_samples`` caps its rows and falls back
    to ``quantize_config.quant_n_samples`` when left None.
    """
    from swift.dev.components import QuantizeComponent

    component = QuantizeComponent(
        model_config,
        template_config,
        quantize_config,
        dataset_config,
        output_dir=output_dir,
        quant_n_samples=quant_n_samples,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
        exist_ok=exist_ok,
        **quantizer_kwargs)
    return component().path

# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        ConvertConfig,
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        TemplateConfig,
    )


def run_convert(
    model_config: ModelConfig,
    convert_config: ConvertConfig,
    *,
    template_config: Optional[TemplateConfig] = None,
    distributed_config: Optional[DistributedConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional[MoEConfig] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Convert between HF and mcore formats; returns the output directory.

    Direction comes from ``convert_config``: ``to_mcore`` without ``mcore_model`` is HF -> mcore,
    ``to_hf`` is mcore -> HF, ``to_mcore`` with ``mcore_model`` is an mcore reshard (and/or
    ``mcore_adapter`` merge). ``output_dir`` overrides ``checkpoint_config.output_dir``.
    """
    from swift.dev.components import ConvertComponent

    component = ConvertComponent(
        model_config,
        convert_config,
        template_config=template_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        output_dir=output_dir)
    return component().path

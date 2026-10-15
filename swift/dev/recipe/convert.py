"""The ``run_convert`` entry point for HF <-> Megatron/mcore weight conversion.

``run_convert`` is the name the CLI (``swift export --to_mcore / --to_hf``) calls; the conversion itself
lives in :class:`swift.dev.components.ConvertComponent`. This module keeps the procedural signature and
returns a plain ``str``, so the CLI is unaffected by the component's typed artifact.
"""
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

    Direction is taken from ``convert_config``:
      - ``to_mcore=True`` with no ``mcore_model``  -> HF  -> mcore
      - ``to_hf=True``                             -> mcore -> HF
      - ``to_mcore=True`` with ``mcore_model``     -> mcore -> mcore (reshard, and/or merge
        ``mcore_adapter``)

    ``output_dir`` overrides ``checkpoint_config.output_dir``.
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

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Recipe-local references to the existing atomic configurations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RewardConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )


@dataclass(kw_only=True)
class RecipeConfigs:
    model: ModelConfig
    template: TemplateConfig
    dataset: DatasetConfig
    train: TrainConfig
    distributed: DistributedConfig
    checkpoint: Optional[CheckpointConfig] = None
    tuner: Optional[TunerConfig] = None
    logging: Optional[LoggingConfig] = None
    quantize: Optional[QuantizeConfig] = None
    megatron: Optional[MegatronConfig] = None
    moe: Optional[MoEConfig] = None
    rl: Optional[RLConfig] = None
    reward: Optional[RewardConfig] = None
    rollout: Optional[RolloutConfig] = None
    generation: Optional[GenerationConfig] = None

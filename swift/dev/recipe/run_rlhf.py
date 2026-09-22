"""Unified RLHF dispatcher for the public CLI and programmatic callers."""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

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
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

_OFFLINE_TYPES = frozenset({'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'})


def run_rlhf(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rollout_config: RolloutConfig,
    rlhf_config: RLHFConfig,
    tuner_config: Optional[TunerConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
    logging_config: Optional[LoggingConfig] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional[MoEConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Dispatch every supported alignment algorithm to its native dev recipe."""
    from swift.dev.recipe.run_dpo import run_dpo
    from swift.dev.recipe.run_gkd import run_gkd
    from swift.dev.recipe.run_grpo import run_grpo
    from swift.dev.recipe.run_ppo import run_ppo

    common = {
        'tuner_config': tuner_config,
        'logging_config': logging_config,
        'quantize_config': quantize_config,
        'megatron_config': megatron_config,
        'moe_config': moe_config,
        'output_dir': output_dir,
        '_save_final': _save_final,
    }
    if rlhf_config.rlhf_type in _OFFLINE_TYPES:
        return run_dpo(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rlhf_config,
            **common,
        )
    if rlhf_config.rlhf_type == 'grpo':
        return run_grpo(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rollout_config,
            rlhf_config,
            generation_config=generation_config,
            **common,
        )
    if rlhf_config.rlhf_type == 'ppo':
        return run_ppo(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rollout_config,
            rlhf_config,
            generation_config=generation_config,
            **common,
        )
    if rlhf_config.rlhf_type == 'gkd':
        return run_gkd(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rlhf_config,
            generation_config=generation_config,
            rollout_config=rollout_config,
            **common,
        )
    raise ValueError(f'Unsupported rlhf_type: {rlhf_config.rlhf_type!r}.')

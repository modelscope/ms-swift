"""Unified RL dispatcher for the public CLI and programmatic callers."""
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
        RewardConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

_OFFLINE_TYPES = frozenset({'dpo', 'kto', 'cpo', 'orpo', 'simpo', 'rm'})


def run_rl(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rollout_config: RolloutConfig,
    rl_config: RLConfig,
    reward_config: Optional[RewardConfig] = None,
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
    from swift.dev.recipe.preference.run_dpo import run_dpo
    from swift.dev.recipe.distill.run_gkd import run_gkd
    from swift.dev.recipe.grpo.run_grpo import run_grpo
    from swift.dev.recipe.distill.run_mopd import run_mopd
    from swift.dev.recipe.distill.run_opsd import run_opsd
    from swift.dev.recipe.ppo.run_ppo import run_ppo
    from swift.dev.recipe.rft.run_rft import run_rft

    common = {
        'tuner_config': tuner_config,
        'logging_config': logging_config,
        'quantize_config': quantize_config,
        'megatron_config': megatron_config,
        'moe_config': moe_config,
        'output_dir': output_dir,
        '_save_final': _save_final,
    }
    if rl_config.rl_type in _OFFLINE_TYPES:
        return run_dpo(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rl_config,
            reward_config=reward_config,
            **common,
        )
    if rl_config.rl_type == 'grpo':
        return run_grpo(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rollout_config,
            rl_config,
            reward_config=reward_config,
            generation_config=generation_config,
            **common,
        )
    if rl_config.rl_type == 'rft':
        return run_rft(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rollout_config,
            rl_config,
            reward_config=reward_config,
            generation_config=generation_config,
            **common,
        )
    if rl_config.rl_type == 'ppo':
        return run_ppo(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rollout_config,
            rl_config,
            reward_config=reward_config,
            generation_config=generation_config,
            **common,
        )
    if rl_config.rl_type == 'gkd':
        return run_gkd(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rollout_config,
            rl_config,
            reward_config=reward_config,
            generation_config=generation_config,
            **common,
        )
    if rl_config.rl_type == 'opsd':
        return run_opsd(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rollout_config,
            rl_config,
            reward_config=reward_config,
            generation_config=generation_config,
            **common,
        )
    if rl_config.rl_type == 'mopd':
        return run_mopd(
            model_config,
            template_config,
            dataset_config,
            train_config,
            distributed_config,
            checkpoint_config,
            rollout_config,
            rl_config,
            reward_config=reward_config,
            generation_config=generation_config,
            **common,
        )
    raise ValueError(f'Unsupported rl_type: {rl_config.rl_type!r}.')

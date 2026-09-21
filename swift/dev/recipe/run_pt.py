"""Pretraining recipe built on the common causal-language-model assembly."""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )


def run_pt(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    tuner_config: Optional[TunerConfig] = None,
    logging_config: Optional[LoggingConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Run causal-LM pretraining while enforcing the non-chat loss contract."""
    if model_config.task_type not in (None, 'causal_lm'):
        raise ValueError(f'PT requires task_type="causal_lm", got {model_config.task_type!r}.')
    if template_config.use_chat_template not in (None, False):
        raise ValueError('PT requires use_chat_template=false.')
    if template_config.loss_scale not in ('default', 'all'):
        raise ValueError('PT requires loss_scale="all".')
    model_config.task_type = 'causal_lm'
    template_config.use_chat_template = False
    template_config.loss_scale = 'all'

    from swift.dev.recipe.run_sft import run_sft
    return run_sft(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        output_dir=output_dir,
        _save_final=_save_final,
    )

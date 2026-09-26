"""SFT CLI: argv -> dev Configs -> run_sft.

dev counterpart of legacy ``swift sft``. The Config dataclasses ARE the argument surface: argv is
parsed straight into them (see ``swift.dev.cli.parser``), so there is no legacy ``SftArguments`` bridge
and no same-name copy step. The eight Configs SFT needs share no field name, so a single flat argparse
namespace has no collisions.

``LoggingConfig`` is parsed alongside the seven core Configs and reaches the common dev tracker used by
both transformers and Megatron training loops.
``GenerationConfig`` stays outside the parse set because its training-time fields are already owned by
``TrainConfig``; unsupported generation flags land in ``remaining`` and are refused.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Tuple

from .legacy_coverage import TRAINING_GENERATION_UNSUPPORTED_FIELDS, TRAIN_UNSUPPORTED_FIELDS, TUNER_UNSUPPORTED_FIELDS

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

# Exhaustive classification of the legacy SftArguments fields which have no dev Config field.
# Precision aliases are normalized centrally by parser.normalize_argv; every remaining field is
# rejected with its category rather than disappearing in a generic same-name copy.
LEGACY_ONLY_CLASSIFICATION = {
    'unsupported_training': TRAIN_UNSUPPORTED_FIELDS + TUNER_UNSUPPORTED_FIELDS + TRAINING_GENERATION_UNSUPPORTED_FIELDS,
}


def parse_sft_configs(
    argv: Optional[List[str]] = None,
    *,
    command: str = 'sft',
) -> Tuple['ModelConfig', 'PluginConfig', 'TemplateConfig', 'DatasetConfig', 'TrainConfig', 'DistributedConfig',
           'CheckpointConfig', 'LoggingConfig', Optional['TunerConfig'], 'QuantizeConfig']:
    """Parse argv into the Configs run_sft consumes; dispatch the tuner by ``tuner_type``.

    Returns tuner_config=None for full-parameter training (tuner_type='full') and a filled TunerConfig
    for 'lora'. Any other tuner_type is refused rather than silently building LoRA.
    """
    from swift.dev.cli.parser import parse_configs, resolve_argv, select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    reject_legacy_only_flags(command, effective_argv)

    classes = [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig,
        LoggingConfig, TunerConfig, QuantizeConfig
    ]
    configs, remaining = parse_configs(classes, effective_argv)
    if remaining:
        raise ValueError(f'Unrecognized arguments: {remaining}. The dev SFT CLI parses the Config surface '
                         'directly; a flag with no matching Config field is refused rather than dropped.')
    (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
     checkpoint_config, logging_config, tuner, quantize_config) = configs

    return (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
            checkpoint_config, logging_config, select_tuner(tuner), quantize_config)


def sft_main(argv: Optional[List[str]] = None) -> List[dict]:
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_embedding, run_reranker, run_seq_cls, run_sft

    (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
     checkpoint_config, logging_config, tuner_config, quantize_config) = parse_sft_configs(argv)
    process_and_validate_configs({
        'model_config': model_config,
        'plugin_config': plugin_config,
        'template_config': template_config,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'checkpoint_config': checkpoint_config,
        'logging_config': logging_config,
        'tuner_config': tuner_config,
        'quantize_config': quantize_config,
    })

    task_type = model_config.task_type or 'causal_lm'
    recipes = {
        'causal_lm': run_sft,
        'seq_cls': run_seq_cls,
        'embedding': run_embedding,
        'reranker': run_reranker,
        'generative_reranker': run_reranker,
    }
    if task_type not in recipes:
        raise ValueError(f'Unsupported SFT task_type: {task_type!r}.')
    return recipes[task_type](
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        quantize_config=quantize_config,
        output_dir=checkpoint_config.output_dir,
    )


if __name__ == '__main__':
    sft_main()

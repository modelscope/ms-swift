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
from typing import Any, Dict, List, Optional

from .legacy_coverage import TRAINING_GENERATION_UNSUPPORTED_FIELDS, TRAIN_UNSUPPORTED_FIELDS, TUNER_UNSUPPORTED_FIELDS

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
) -> Dict[str, Any]:
    """Parse argv into the Configs run_sft/run_pt consume; dispatch the tuner by ``tuner_type``.

    The backend is a single flag: ``--backend megatron`` selects the Megatron path (and its own
    legacy-flag contract, keyed ``megatron_<command>``), while the default Transformers path is
    unchanged. MegatronConfig/MoEConfig are always parsed so the Megatron knobs are expressible; on
    the Transformers backend validate_configs rejects any of them that were actually set.

    ``tuner_config`` is None for full-parameter training (tuner_type='full') and a filled TunerConfig
    for 'lora'. Any other tuner_type is refused rather than silently building LoRA.
    """
    from swift.dev.cli._megatron_compat import MegatronCliCompatConfig, configure_megatron, is_megatron_argv
    from swift.dev.cli.parser import parse_configs, resolve_argv, select_tuner
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        PluginConfig,
        QuantizeConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    is_megatron = is_megatron_argv(effective_argv)
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    reject_legacy_only_flags(f'megatron_{command}' if is_megatron else command, effective_argv)

    classes = [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig,
        LoggingConfig, TunerConfig, QuantizeConfig, MegatronConfig, MoEConfig
    ]
    if is_megatron:
        classes.append(MegatronCliCompatConfig)
    configs, remaining = parse_configs(classes, effective_argv)
    if remaining:
        raise ValueError(f'Unrecognized arguments: {remaining}. The dev SFT CLI parses the Config surface '
                         'directly; a flag with no matching Config field is refused rather than dropped.')
    (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
     checkpoint_config, logging_config, tuner, quantize_config, megatron_config, moe_config) = configs[:12]

    if is_megatron:
        tuner = configure_megatron(model_config, train_config, distributed_config, tuner)
    else:
        tuner = select_tuner(tuner)

    return {
        'model_config': model_config,
        'plugin_config': plugin_config,
        'template_config': template_config,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'checkpoint_config': checkpoint_config,
        'logging_config': logging_config,
        'tuner_config': tuner,
        'quantize_config': quantize_config,
        'megatron_config': megatron_config,
        'moe_config': moe_config,
    }


def sft_main(argv: Optional[List[str]] = None) -> List[dict]:
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_embedding, run_reranker, run_seq_cls, run_sft

    configs = parse_sft_configs(argv)
    process_and_validate_configs(configs)

    model_config = configs['model_config']
    distributed_config = configs['distributed_config']
    is_megatron = distributed_config.backend == 'megatron'
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
    if is_megatron and task_type != 'causal_lm':
        raise ValueError(f'--backend megatron supports only causal_lm SFT, got task_type={task_type!r}. '
                         'Drop --backend to train seq_cls/embedding/reranker on the transformers backend.')
    # Only run_sft/run_pt/run_rlhf accept the Megatron Configs; the other task types are HF-only.
    megatron_kwargs = ({
        'megatron_config': configs['megatron_config'],
        'moe_config': configs['moe_config']
    } if is_megatron else {})
    return recipes[task_type](
        model_config,
        configs['template_config'],
        configs['dataset_config'],
        configs['train_config'],
        distributed_config=distributed_config,
        checkpoint_config=configs['checkpoint_config'],
        tuner_config=configs['tuner_config'],
        logging_config=configs['logging_config'],
        quantize_config=configs['quantize_config'],
        output_dir=configs['checkpoint_config'].output_dir,
        **megatron_kwargs,
    )


if __name__ == '__main__':
    sft_main()
